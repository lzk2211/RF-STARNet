from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
# from torch.cuda import amp
import torch

from tqdm import tqdm

import sys, select, time, os

# from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

from confusionMatrix import ConfusionMatrix

# from syops import get_model_complexity_info
from datetime import datetime
from tabulate import tabulate
# 延迟导入 sklearn.metrics、matplotlib、pandas、seaborn 到使用函数内部，避免导入期触发 ABI 问题
import numpy as np

def normalize(matrix):
    # 假设输入矩阵的形状为 (batch_size, height, width)
    batch_size = matrix.shape[0]
    normalized_matrix = torch.empty_like(matrix)
    
    for i in range(batch_size):
        min_val = torch.min(matrix[i])
        max_val = torch.max(matrix[i])
        normalized_matrix[i] = (matrix[i] - min_val) / (max_val - min_val)
    
    return normalized_matrix

# Removed multi-label helper (compute_adaptive_thresholds)

class CombinedLoss(nn.Module):
    """
    实现论文中的组合损失函数 Gamma_Loss 。
    Gamma_Loss = L_CE + L_ETF
    
    - L_CE: 标准交叉熵损失 
    - L_ETF: 特征-权重对齐损失 (基于 [cite: 306-311, 706])
    """
    def __init__(self, classifier_weights, alpha=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 归一化并存储分类器权重 (码本) [cite: 308]
        # classifier_weights (C) 的形状为 [feature_dim, num_classes]
        norm_weights = F.normalize(classifier_weights, p=2, dim=0)
        self.register_buffer('norm_weights', norm_weights)

    def forward(self, logits, features, labels):
        """
        logits: 模型的原始输出 [batch_size, num_classes]
        features: 展平的特征向量 (x_bar) [batch_size, feature_dim]
        labels: 真实标签 [batch_size]
        """
        
        # 1. L_CE (Cross-Entropy Loss) 
        loss_ce = self.ce_loss(logits, labels)
        
        # 2. L_ETF (Feature-Weight Alignment Loss)
        #    我们将其实现为归一化特征与归一化目标权重间的MSE
        
        # 归一化特征 (码字) [cite: 307]
        norm_features = F.normalize(features, p=2, dim=1)

        # 处理 labels：支持 class index 或 one-hot 格式，确保为 long 索引
        if labels.dim() > 1 and labels.size(-1) > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels.view(-1)
        labels_idx = labels_idx.long()
        # 确保索引用的 device 与 norm_weights 相同，避免跨设备索引错误
        labels_idx = labels_idx.to(self.norm_weights.device)

        # 获取批次中每个样本对应的目标权重向量
        # self.norm_weights.T 的形状是 [num_classes, feature_dim]
        target_weights = self.norm_weights.T[labels_idx]
        
        # 计算 L_ETF (L_ETF_bar [cite: 320])
        loss_etf = F.mse_loss(norm_features, target_weights)
        
        # 3. Total Loss (Gamma_Loss) 
        total_loss = loss_ce + self.alpha * loss_etf
        
        return total_loss
def train(args, folder_path, model, device, train_loader, validate_loader):
    if sys.platform.startswith('linux'):
        system=0
    elif sys.platform.startswith('win'):
        system=1
        import msvcrt

    if args.resume:
        # 加载检查点并继续训练
        checkpoint = torch.load('checkpoint_epoch_10.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0
    
    # scaler = amp.GradScaler()
    print(f"Training with {args.epochs} epochs, batch size {args.batch_size}, learning rate {args.lr}")
    writer = SummaryWriter(folder_path + '/runs')    # Define loss function and optimizer

    # 单标签损失与优化器
    loss_function = nn.CrossEntropyLoss()
    loss_fn = CombinedLoss(
        classifier_weights=model.classifier.weight, 
        alpha=1.0 # 损失的加权因子
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_acc = 0.0
    save_path = folder_path + '/model.pth'
    train_steps = len(train_loader)

    start_time = time.time()

    # Training model
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        # Track training accuracy
        train_correct = 0
        train_total = 0
        for step, data in enumerate(train_bar):
            freq_seq, labels, _ = data
            # img = torch.cat((time_seq, freq_seq), dim=-1).to(device)
            optimizer.zero_grad()
            freq_seq = freq_seq.to(device)
            labels = labels.to(device)

            logits, features = model(freq_seq)

            # Support both class-index labels and one-hot labels
            if labels.dim() > 1 and labels.size(-1) > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels.view(-1)

            loss = loss_fn(logits, features, labels)

            total_loss = loss
            total_loss.backward()
            # loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.item()

            # accuracy
            _, preds = torch.max(logits, dim=1)
            batch_correct = (preds == labels_idx).sum().item()
            train_correct += batch_correct
            train_total += labels_idx.size(0)

            # Print statistics (show loss and running accuracy)
            running_acc = train_correct / train_total if train_total > 0 else 0.0
            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(
                epoch + 1, args.epochs, loss.item(), running_acc)

        # average train loss over steps
        train_loss = train_loss / train_steps
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        # Validation phase

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.eval()
        val_acc = 0
        all_outputs = []
        all_labels = []
        snrs = []
        snr_datas = {"-20dB": 0, "-15dB": 0, "-10dB": 0, "-5dB": 0, "0dB": 0, "5dB": 0, "pure": 0}
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                freq_seq, val_labels, snr = val_data
                freq_seq = freq_seq.to(device)
                val_labels = val_labels.to(device)

                outputs, features = model(freq_seq)
                # outputs = model(torch.real(freq_seq), torch.imag(freq_seq))

                all_outputs.append(outputs.cpu())
                all_labels.append(val_labels.cpu())
                snrs.extend(snr)

        # 合并所有 batch
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 单标签分类，使用softmax输出
        _, predicted = torch.max(all_outputs, dim=1)
        # all_labels = torch.argmax(all_labels, dim=1)
        correct = (predicted == all_labels).float()
        batch_acc = correct.sum().item()

        val_acc = batch_acc
        val_accurate = val_acc / len(validate_loader.dataset)

        for i, corr in enumerate(correct):
            snr_datas[snrs[i]] += 1 if corr.item() == 1 else 0
        for key, value in snr_datas.items():
            snr_datas[key] = value / snrs.count(key)
        
        summary = f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accurate:.4f} | -20dB: {snr_datas['-20dB']:.4f} | -15dB: {snr_datas['-15dB']:.4f} | -10dB: {snr_datas['-10dB']:.4f} | -5dB: {snr_datas['-5dB']:.4f} | 0dB: {snr_datas['0dB']:.4f} | 5dB: {snr_datas['5dB']:.4f} | pure: {snr_datas['pure']:.4f} | LR: {current_lr:.6f}"
        print(summary)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), save_path)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_accuracy, epoch)
        writer.add_scalar('Acc/val', val_accurate, epoch)

        if True:# 按键Q回车提前退出
            if system == 1 and msvcrt.kbhit():  # 检查是否有键盘输入
                user_input = msvcrt.getch().decode('utf-8').strip()
                if user_input == 'q':
                    print("Stopping training after this epoch.")
                    break
            elif system == 0 and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:  # Check if 'q' was pressed
                user_input = sys.stdin.readline().strip()
                if user_input == 'q':
                    print("Stopping training after this epoch.")
                    break

    print('Finished Training')

    stop_time = time.time()
    print(f'Training time: {stop_time - start_time:.2f} seconds')

    writer.close()
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), folder_path + '/last_model.pth')


def test(args, folder_path, model, device, test_loader, target_classes):
    print('Testing')
    confusion = ConfusionMatrix(num_classes=args.class_num, labels=target_classes)

    model_path = folder_path + '/model.pth'
    assert os.path.exists(model_path), "cannot find {} file".format(model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    model.to(device)
    
    model.eval()
    all_outputs = []
    all_labels = []
    snrs = []
    snr_datas = {"-20dB": 0, "-15dB": 0, "-10dB": 0, "-5dB": 0, "0dB": 0, "5dB": 0, "pure": 0}

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            freq_seq, test_labels, snr = test_data
            freq_seq = freq_seq.to(device)
            test_labels = test_labels.to(device)

            outputs, features = model(freq_seq)
            outputs_softmax = torch.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs_softmax, dim=1)
            # labels_idx = torch.argmax(test_labels, dim=1)
            confusion.update(outputs_argmax.to("cpu").numpy(), test_labels.to("cpu").numpy())

            all_outputs.append(outputs.cpu())
            all_labels.append(test_labels.cpu())
            snrs.extend(snr)

    # 合并所有 batch
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # 单标签分类，使用softmax输出
    _, predicted = torch.max(all_outputs, dim=1)
    # all_labels = torch.argmax(all_labels, dim=1)
    correct = (predicted == all_labels).float()
    batch_acc = correct.sum().item()

    confusion.plot(folder_path)
    acc, table = confusion.summary()

    test_acc = batch_acc
    test_accurate = test_acc / len(test_loader.dataset)

    for i, corr in enumerate(correct):
        snr_datas[snrs[i]] += 1 if corr.item() == 1 else 0
    for key, value in snr_datas.items():
        snr_datas[key] = value / snrs.count(key)

    print('[Test] Accuracy: %.3f' % test_accurate)
    summary = f"[Test] Avg Accuracy: {test_accurate:.4f} | -20dB: {snr_datas['-20dB']:.4f} | -15dB: {snr_datas['-15dB']:.4f} | -10dB: {snr_datas['-10dB']:.4f} | -5dB: {snr_datas['-5dB']:.4f} | 0dB: {snr_datas['0dB']:.4f} | 5dB: {snr_datas['5dB']:.4f} | pure: {snr_datas['pure']:.4f}"
    print(summary)

    # precision, recall, f1, macro_ap, micro_ap = plot_curve(args, predicted, all_labels, folder_path)

    return test_accurate#, precision, recall, f1, macro_ap, micro_ap


from thop import profile
from collections import defaultdict  # NEW: 用于构建嵌套字典
from sklearn.metrics import precision_recall_fscore_support

def test(args, folder_path, model, device, test_loader, target_classes):
    print('Testing')
    confusion = ConfusionMatrix(num_classes=args.class_num, labels=target_classes)

    model_path = folder_path + '/model.pth'
    assert os.path.exists(model_path), "cannot find {} file".format(model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    model.to(device)
    model.eval()
    
    all_outputs = []
    all_labels = []
    
    # NEW: 创建一个字典来按SNR存储预测和标签
    # 结构: {'-20dB': {'preds': [...], 'labels': [...]}, '-15dB': ...}
    snr_results = defaultdict(lambda: {'preds': [], 'labels': []})
    total_infer_start = time.time()
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            freq_seq, test_labels, snr_batch = test_data  # 'snr' 重命名为 'snr_batch' 避免与列表混淆
            freq_seq = freq_seq.to(device)
            test_labels = test_labels.to(device)

            outputs, feature = model(freq_seq)
            outputs_softmax = torch.softmax(outputs, dim=1)
            outputs_argmax = torch.argmax(outputs_softmax, dim=1)
            
            # NEW: 将CPU上的结果存入 snr_results 字典
            batch_preds = outputs_argmax.cpu().numpy()
            batch_labels = test_labels.cpu().numpy()
            
            for i in range(len(batch_labels)):
                current_snr = snr_batch[i]
                snr_results[current_snr]['preds'].append(batch_preds[i])
                snr_results[current_snr]['labels'].append(batch_labels[i])

            # (这部分保持不变，用于计算总体混淆矩阵和总准确率)
            confusion.update(batch_preds, batch_labels)
            all_outputs.append(outputs.cpu())
            all_labels.append(test_labels.cpu())

    # --- 循环后 ---
    
    # --- 循环后，统计推理时间与显存 ---
    if torch.cuda.is_available() and (device.type == 'cuda' or 'cuda' in str(device)):
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)
        cur_mem = torch.cuda.memory_allocated(device)
    else:
        peak_mem = 0
        cur_mem = 0
    total_infer_sec = time.time() - total_infer_start
    
    # (这部分保持不变，用于计算总体混淆矩阵和总准确率)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    _, predicted = torch.max(all_outputs, dim=1)
    correct = (predicted == all_labels).float()
    batch_acc = correct.sum().item()
    # Overall precision/recall/F1 (macro average)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels.cpu().numpy(),
        predicted.cpu().numpy(),
        average='macro',
        zero_division=0
    )

    confusion.plot(folder_path)
    acc, table = confusion.summary()

    test_acc = batch_acc
    test_accurate = test_acc / len(test_loader.dataset)
    print('[Test] Overall Accuracy: %.3f' % test_accurate)
    # 统计吞吐与GPU小时（按单 GPU 计）
    total_samples = len(test_loader.dataset)
    samples_per_sec = (total_samples / total_infer_sec) if total_infer_sec > 0 else 0.0
    gpu_hours = total_infer_sec / 3600.0
    samples_per_gpu_hour = total_samples / gpu_hours if gpu_hours > 0 else 0.0
    # 显存 MB
    mem_mb = cur_mem / (1024**2)
    peak_mb = peak_mem / (1024**2)
    avg_ms_per_sample = (total_infer_sec / total_samples * 1000.0) if total_samples > 0 else 0.0
    print(f"[Test] Inference time: {total_infer_sec:.3f}s | {avg_ms_per_sample:.3f} ms/sample | Throughput: {samples_per_sec:.2f} samples/s")
    print(f"[Test] GPU memory: current {mem_mb:.1f} MB | peak {peak_mb:.1f} MB | GPU-hours: {gpu_hours:.4f} h | samples/GPU-hour: {samples_per_gpu_hour:.1f}")

    # --- NEW: 按SNR计算详细指标 (替换您原有的 snr_datas 循环) ---
    
    print("\n--- Per-SNR Metrics Summary ---")
    print(f"{'SNR':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 54)

    snr_levels = ['-20dB', '-15dB', '-10dB', '-5dB', '0dB', '5dB', 'pure']
    snr_accuracy_data = {}   # 每个 SNR 的准确率
    snr_precision_data = {}  # 每个 SNR 的宏平均精确率
    snr_recall_data = {}     # 每个 SNR 的宏平均召回率
    snr_f1_data = {}         # 每个 SNR 的宏平均 F1

    for snr_val in snr_levels:
        if snr_val not in snr_results:
            print(f"Warning: No data found for SNR {snr_val}")
            snr_accuracy_data[snr_val] = 0
            snr_precision_data[snr_val] = 0
            snr_recall_data[snr_val] = 0
            snr_f1_data[snr_val] = 0
            continue
            
        snr_preds = snr_results[snr_val]['preds']
        snr_labels = snr_results[snr_val]['labels']
        
        if len(snr_labels) == 0:
            print(f"Warning: No samples for SNR {snr_val}")
            snr_accuracy_data[snr_val] = 0
            snr_precision_data[snr_val] = 0
            snr_recall_data[snr_val] = 0
            snr_f1_data[snr_val] = 0
            continue

        # 使用 sklearn.metrics.precision_recall_fscore_support
        # average='macro': 计算每个类的指标，然后取平均值，不考虑类别不平衡
        # zero_division=0: 避免在分母为0时（例如某个类从未被预测）打印警告
        precision, recall, f1, _ = precision_recall_fscore_support(
            snr_labels, 
            snr_preds, 
            average='macro', 
            zero_division=0
        )
        
        # 单独计算准确率
        correct_predictions = sum(p == l for p, l in zip(snr_preds, snr_labels))
        accuracy = correct_predictions / len(snr_labels)

        # 存储各指标
        snr_accuracy_data[snr_val] = accuracy
        snr_precision_data[snr_val] = precision
        snr_recall_data[snr_val] = recall
        snr_f1_data[snr_val] = f1

        # 打印新表格
        print(f"{snr_val:<8} | {accuracy:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

    print("--- End of Summary ---")

    # MATLAB-friendly strings
    values_acc = ', '.join(f"{snr_accuracy_data[k]:.4f}" for k in snr_levels)
    values_prec = ', '.join(f"{snr_precision_data[k]:.4f}" for k in snr_levels)
    values_rec = ', '.join(f"{snr_recall_data[k]:.4f}" for k in snr_levels)
    values_f1 = ', '.join(f"{snr_f1_data[k]:.4f}" for k in snr_levels)

    # --- 计算 FLOPs 和 Params ---
    # 注意：这里假设输入尺寸为 (1, 1, 512, 512)，如果您的模型输入是 (1, 2, 1024) 等其他形状，请修改此处
    try:
        # 尝试获取一个真实的 batch 来确定输入维度，更稳健
        dummy_input = next(iter(test_loader))[0][0:1].to(device)
    except:
        # 如果失败则使用默认形状
        dummy_input = torch.randn(1, 1, 512, 512).to(device)

    flops, params = profile(model, (dummy_input,), verbose=False)
    flops_int = int(flops)
    params_int = int(params)

    # --- 计算平均单样本显存占用 ---
    # 显存占用是按 Batch 分配的，所以单样本显存 = 峰值显存 / Batch Size
    current_batch_size = test_loader.batch_size
    mem_per_sample_MB = peak_mb / current_batch_size if current_batch_size > 0 else 0.0

    # --- 构造最终输出字典字符串 ---
    # 完全匹配您要求的格式
    metrics_struct = (
        f"{args.model}_metrics = {{"
        f"'avg_acc': {test_accurate:.4f}, 'avg_precision': {overall_precision:.4f}, "
        f"'avg_recall': {overall_recall:.4f}, 'avg_f1': {overall_f1:.4f}, "
        f"'acc': [{values_acc}], 'precision': [{values_prec}], "
        f"'recall': [{values_rec}], 'f1': [{values_f1}], "
        f"'flops': {flops_int}, 'params': {params_int}, "
        f"'infer_sec': {total_infer_sec:.6f}, "
        # f"'infer_mem_MB': {mem_mb:.2f}, "         # 当前显存
        # f"'infer_peak_MB': {peak_mb:.2f}, "       # 峰值显存 (Batch级别)
        f"'mem_per_sample_MB': {mem_per_sample_MB:.4f}, " # 平均单样本显存
        f"'samples_per_sec': {samples_per_sec:.4f}}}"
        # f"'gpu_hours': {gpu_hours:.6f}, "
        # f"'samples_per_gpu_hour': {samples_per_gpu_hour:.2f}}}"
    )

    print("\n" + "="*20 + " Final Metrics for Copying " + "="*20)
    print(metrics_struct)
    print("="*65)

    # model_name_metrics = {'acc': [...], ...} 这一行可以直接复制到 Python 或 MATLAB 中使用

    return test_accurate