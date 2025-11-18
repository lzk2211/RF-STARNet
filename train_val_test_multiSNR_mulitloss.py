from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.nn.functional as F
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
from thop import profile

from collections import defaultdict  # NEW: 用于构建嵌套字典
from sklearn.metrics import precision_recall_fscore_support  # NEW: 导入计算指标的工具

from DA import *

def normalize(matrix):
    # 假设输入矩阵的形状为 (batch_size, height, width)
    batch_size = matrix.shape[0]
    normalized_matrix = torch.empty_like(matrix)
    
    for i in range(batch_size):
        min_val = torch.min(matrix[i])
        max_val = torch.max(matrix[i])
        normalized_matrix[i] = (matrix[i] - min_val) / (max_val - min_val)
    
    return normalized_matrix

def orthogonality_loss(feat1, feat2):
    # 归一化
    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)
    cross_cov = torch.mm(feat1_norm.t(), feat2_norm)
    # 如果 feat1 和 feat2 维度不同，需要先投影到相同维度，或者只计算相关性矩阵
    # 这里假设你已经投影或者维度不同直接计算相关性矩阵的范数
    # 简单做法：计算余弦相似度的绝对值平均
    # 注意：你的 res_feat (128) 和 sp_feat (512) 维度不同，需要先对齐
    # 建议：只对 batch 内的样本相关性做约束，或者在 gate 之后的特征做约束
    
    return torch.mean(torch.abs(cross_cov))

# Removed multi-label helper (compute_adaptive_thresholds)

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        train_loss_res = 0
        train_loss_sp = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        # Track training accuracy
        train_correct = 0
        train_total = 0
        for step, data in enumerate(train_bar):
            inputs, labels, snr = data
            # img = torch.cat((time_seq, inputs), dim=-1).to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # inputs = spec_augment(inputs)
            # inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, snr)

            main, logits_res, logits_sp, res_feat, sp_feat = model(inputs)

            # Support both class-index labels and one-hot labels
            if labels.dim() > 1 and labels.size(-1) > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels.view(-1)

            # loss = mixup_criterion(loss_function, main, targets_a, targets_b, lam)
            loss = loss_function(main, labels_idx)
            loss_res = loss_function(logits_res, labels_idx)
            loss_sp = loss_function(logits_sp, labels_idx)
            loss_ortho = orthogonality_loss(res_feat, sp_feat)# new loss
            alpha1 = 0.5
            alpha2 = 0.5
            alpha3 = 0.1
            total_loss = loss + alpha1 * loss_res + alpha2 * loss_sp + alpha3 * loss_ortho

            total_loss = total_loss
            total_loss.backward()
            # loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.item()
            train_loss_res += loss_res.item()
            train_loss_sp += loss_sp.item()

            # accuracy
            _, preds = torch.max(main, dim=1)
            batch_correct = (preds == labels_idx).sum().item()
            train_correct += batch_correct
            train_total += labels_idx.size(0)

            # Print statistics (show loss and running accuracy)
            running_acc = train_correct / train_total if train_total > 0 else 0.0
            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(
                epoch + 1, args.epochs, loss.item(), running_acc)

        # average train loss over steps
        train_loss = train_loss / train_steps
        train_loss_res = train_loss_res / train_steps
        train_loss_sp = train_loss_sp / train_steps
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        # Validation phase

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.eval()
        val_acc = 0
        all_outputs = []
        all_labels = []
        snrs = []
        # val_steps = len(validate_loader)

        snr_datas = {"-20dB": 0, "-15dB": 0, "-10dB": 0, "-5dB": 0, "0dB": 0, "5dB": 0, "pure": 0}
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                freq_seq, val_labels, snr = val_data
                freq_seq = freq_seq.to(device)
                val_labels = val_labels.to(device)

                main = model(freq_seq)
                # outputs = model(torch.real(freq_seq), torch.imag(freq_seq))

                all_outputs.append(main.cpu())
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
        writer.add_scalar('Loss_res/train', train_loss_res, epoch)
        writer.add_scalar('Loss_sp/train', train_loss_sp, epoch)
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
    
    # NEW: 创建一个字典来按SNR存储预测和标签
    # 结构: {'-20dB': {'preds': [...], 'labels': [...]}, '-15dB': ...}
    snr_results = defaultdict(lambda: {'preds': [], 'labels': []})
    
    # 推理计时与显存统计准备
    total_infer_start = time.time()
    if torch.cuda.is_available() and (device.type == 'cuda' or 'cuda' in str(device)):
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            iter_start = time.time()
            freq_seq, test_labels, snr_batch = test_data  # 'snr' 重命名为 'snr_batch' 避免与列表混淆
            freq_seq = freq_seq.to(device)
            test_labels = test_labels.to(device)

            main = model(freq_seq)
            outputs_softmax = torch.softmax(main, dim=1)
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
            all_outputs.append(main.cpu())
            all_labels.append(test_labels.cpu())

            # 可选：记录每 batch 用时（不做逐条打印，避免刷屏）
            _ = time.time() - iter_start

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

    # --- (这部分使用新的 snr_accuracy_data，保持您原有的输出格式) ---
    
    # MATLAB-friendly one-liners for easy copy
    values_acc = ', '.join(f"{snr_accuracy_data[k]:.4f}" for k in snr_levels)
    values_prec = ', '.join(f"{snr_precision_data[k]:.4f}" for k in snr_levels)
    values_rec = ', '.join(f"{snr_recall_data[k]:.4f}" for k in snr_levels)
    values_f1 = ', '.join(f"{snr_f1_data[k]:.4f}" for k in snr_levels)

    # 计算模型的 FLOPs 和参数量（放在结构体输出前，以便一起打印）
    inputs = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (inputs,), verbose=False)
    flops_int = int(flops)  # 转为整数更便于 MATLAB 处理
    params_int = int(params)

    # MATLAB struct-style dictionary for easy paste (加入 flops/params & 推理统计)
    metrics_struct = (
        f"{args.model}_metrics = {{'acc': [{values_acc}], 'precision': [{values_prec}], "
        f"'recall': [{values_rec}], 'f1': [{values_f1}], 'flops': {flops_int}, 'params': {params_int}, "
        f"'infer_mem_MB': {mem_mb:.2f}, 'infer_peak_MB': {peak_mb:.2f}, "
        f"'samples_per_sec': {samples_per_sec:.4f}, 'gpu_hours': {gpu_hours:.6f}, 'samples_per_gpu_hour': {samples_per_gpu_hour:.2f}}}"
    )
    print(metrics_struct)
    print(f"(model complexity) flops={flops_int}, params={params_int}")

    # precision, recall, f1, macro_ap, micro_ap = plot_curve(args, predicted, all_labels, folder_path)


    return test_accurate#, precision, recall, f1, macro_ap, micro_ap

