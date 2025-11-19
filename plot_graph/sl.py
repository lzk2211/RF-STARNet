import pandas as pd
import numpy as np
from typing import Optional, Tuple

def _try_read(file_path: str, seps=("\t", ",", ";")) -> Tuple[pd.DataFrame, str]:
    """尝试不同分隔符读取，返回成功的 DataFrame 和使用的分隔符。"""
    last_err = None
    for s in seps:
        try:
            df = pd.read_csv(file_path, sep=s)
            # 需要包含 Step 与 Value 两列（大小写兼容）
            cols = {c.lower(): c for c in df.columns}
            if 'step' in cols and 'value' in cols:
                return df, s
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法读取文件，尝试分隔符 {seps} 失败: {last_err}")

def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")

def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """尝试在常见列名中寻找时间列，返回原始列名或 None。"""
    candidates = {c: _normalize_name(c) for c in df.columns}
    for orig, norm in candidates.items():
        if norm in ("walltime", "time", "timestamp"):
            return orig
    return None

def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def analyze_convergence(file_path: str, window_size: int = 3, threshold: float = 0.005) -> Optional[pd.DataFrame]:
    """
    输入: TensorBoard 导出的 metrics CSV，需包含 Step 与 Value 列。
    输出: 打印最佳精度与开始平稳的轮次；返回带 rolling_std 的 DataFrame。
    """
    try:
        df, used_sep = _try_read(file_path)
    except Exception as e:
        print(f"[读取失败] {e}")
        return None

    # 标准化列名空白
    df.columns = [c.strip() for c in df.columns]
    # 统一引用列
    step_col = next(c for c in df.columns if c.lower() == 'step')
    value_col = next(c for c in df.columns if c.lower() == 'value')
    
    # 时间列（必须存在）：优先使用 'Wall time'（规范化比对）
    wall_time_col = next((c for c in df.columns if _normalize_name(c) == 'walltime'), None)
    if wall_time_col is None:
        # 兼容 Time/Timestamp，但如果都没有则报错
        wall_time_col = _find_time_column(df)
    if wall_time_col is None:
        raise RuntimeError("未找到时间列：需要包含 'Wall time' 列或等价时间列")

    # 保留时间/步/值三列，防止把时间列丢掉
    df = df[[wall_time_col, step_col, value_col]].dropna().reset_index(drop=True)

    # 最佳值
    best_idx = df[value_col].idxmax()
    best_step = int(df.loc[best_idx, step_col])
    best_val = float(df.loc[best_idx, value_col])

    print(f"【文件读取成功，使用分隔符: '{used_sep}'】")
    print(f"【基于最高精度判断】验证集 Acc 在第 {best_step} 轮达到最高值: {best_val:.6f}")

    times = pd.to_numeric(df[wall_time_col], errors='coerce')
    if not times.notna().any():
        raise RuntimeError("时间列无法解析为数字")
    t0 = float(times.min())

    # 新逻辑：最大值带宽震荡判定
    # 定义带宽下界：不低于 (vmax - threshold)
    vmax = df[value_col].max()
    lower_bound = vmax - float(threshold)
    vals = df[value_col].to_numpy()

    # 找到最早的索引 i，使得从 i 开始的所有值都 >= lower_bound
    chosen_idx = None
    for i in range(len(vals)):
        if vals[i] >= lower_bound:
            if np.all(vals[i:] >= lower_bound):
                chosen_idx = i
                break

    print(f"【最大值震荡带判定】带宽={threshold} => 下界={lower_bound:.6f}")
    if chosen_idx is not None:
        chosen_step = int(df.loc[chosen_idx, step_col])
        chosen_val = float(df.loc[chosen_idx, value_col])
        # 以最早时间为起点，计算小时
        tc = float(times.iloc[chosen_idx]) if chosen_idx < len(times) and pd.notna(times.iloc[chosen_idx]) else None
        if tc is None:
            raise RuntimeError("无法从时间列读取选中步的时间戳")
        elapsed_hours = (tc - t0) / 3600.0
        print(f"第 {chosen_step} 轮进入最大值带宽内并且之后未跌破下界，选取该点；值为 {chosen_val:.6f}；耗时 {elapsed_hours:.3f} 小时")
    else:
        print("未找到满足‘进入最大值带宽且之后不再跌破’的点；可适当放宽带宽 threshold。")

    return df


if __name__ == '__main__':
    file_path = '/home/kk/lzk/SRSpecNet/plot_graph/csv/F_cnn_tiny_1119_runs.csv'
    analyze_convergence(file_path, window_size=3, threshold=0.1)