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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, verbose=True)

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

            outputs = model(freq_seq)

            # Support both class-index labels and one-hot labels
            if labels.dim() > 1 and labels.size(-1) > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels.view(-1)

            loss = loss_function(outputs, labels_idx)

            total_loss = loss
            total_loss.backward()
            # loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.item()

            # accuracy
            _, preds = torch.max(outputs, dim=1)
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
        # val_steps = len(validate_loader)

        snr_datas = {"-20dB": 0, "-15dB": 0, "-10dB": 0, "-5dB": 0, "0dB": 0, "5dB": 0, "pure": 0}
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                freq_seq, val_labels, snr = val_data
                freq_seq = freq_seq.to(device)
                val_labels = val_labels.to(device)

                outputs = model(freq_seq)
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

            outputs = model(freq_seq)
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

def plot_curve(args, all_probs, all_labels, folder_path):
    # 延迟导入，避免启动时 ABI 问题；仅在需要画 PR 曲线时使用
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    # ====== 画PR曲线 ======
    # ====== 计算PR、精确率、召回率、AP =======
    all_probs = torch.cat(all_probs, dim=0).numpy()   # shape: [样本数, 类别数]
    all_labels = torch.cat(all_labels, dim=0).numpy() # shape: [样本数, 类别数] - 多标签格式    # 每个类别的AP（平均精确率）
    ap_scores = []
    for class_idx in range(args.class_num):
        binary_labels = all_labels[:, class_idx]  # 每个类别的二元标签 (0或1)
        class_probs = all_probs[:, class_idx]     # 每个类别的预测概率
        
        ap_score = average_precision_score(binary_labels, class_probs)
        ap_scores.append(ap_score)

    # 宏平均AP
    macro_ap = np.mean(ap_scores)

    # 微平均AP（对所有样本计算）
    micro_ap = average_precision_score(all_labels.ravel(), all_probs.ravel())    # 计算多标签情况下的精确率、召回率、F1
    # 将概率转换为二进制预测 (阈值0.5)
    binary_preds = (all_probs >= 0.5).astype(int)
    
    # 使用样本级别的平均计算每个指标
    precision = precision_score(all_labels, binary_preds, average='samples')
    recall = recall_score(all_labels, binary_preds, average='samples')
    f1 = f1_score(all_labels, binary_preds, average='samples')

    # 打印
    print(f"Macro-averaged Precision: {precision:.4f}")
    print(f"Macro-averaged Recall: {recall:.4f}")
    print(f"Macro-averaged F1: {f1:.4f}")
    print(f"Macro-averaged AP: {macro_ap:.4f}")
    print(f"Micro-averaged AP: {micro_ap:.4f}")

    all_num = precision, recall, f1, macro_ap, micro_ap    # ====== 画PR曲线（多标签）======
    plt.figure()
    for class_idx in range(args.class_num):
        binary_labels = all_labels[:, class_idx]  # 获取第class_idx类的所有标签
        class_probs = all_probs[:, class_idx]    # 获取第class_idx类的所有预测概率

        precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
        ap_score = average_precision_score(binary_labels, class_probs)

        plt.plot(recall, precision, label=f'Class {class_idx} (AP={ap_score:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (per class)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder_path, 'pr_curve_all_classes.png'))
    plt.close()

    print("PR曲线已保存到:", os.path.join(folder_path, 'pr_curve_all_classes.png'))
    return all_num


def test_plot(all_labels, predicted, target_classes, folder_path):
    '''
    画出热图并且输出各类别的准确率
    all_labels: 真实标签，shape: [样本数, 类别数]
    predicted: 模型预测标签，shape: [样本数, 类别数]
    target_classes: 类别名称列表
    folder_path: 保存结果的文件夹路径
    '''
    # 计算正确性矩阵：1 = 正确，0 = 错误
    correct_matrix = (all_labels.numpy() == predicted.numpy()).astype(int)

    # 找出有错误的样本（行）
    error_rows_mask = (correct_matrix.sum(axis=1) < correct_matrix.shape[1])
    error_matrix = correct_matrix[error_rows_mask]

    # 构建可视化矩阵：正确为 1，错误为 -1
    vis_matrix = np.where(error_matrix == 1, 1, -1)

    # 画图（延迟导入，若失败则跳过）
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except Exception as e:
        print(f"Skip error heatmap due to plotting deps issue: {e}")
        plt = None

    if plt is not None:
        plt.figure(figsize=(min(20, vis_matrix.shape[1]), max(6, vis_matrix.shape[0] * 0.25)))
        sns.heatmap(vis_matrix, cmap=sns.color_palette(["red", "blue"]), cbar=False,
                    xticklabels=target_classes,
                    yticklabels=[f"S{i}" for i in np.where(error_rows_mask)[0]])
        plt.title("Prediction Correctness Heatmap (Only Samples with Errors)")
        plt.xlabel("Labels")
        plt.ylabel("Sample Index (Only Error Rows)")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "error_samples_heatmap.png"))
    # plt.show()

    true_labels = all_labels.numpy()
    pred_labels = predicted.numpy()

    TP = np.logical_and(true_labels == 1, pred_labels == 1).sum(axis=0)
    FN = np.logical_and(true_labels == 1, pred_labels == 0).sum(axis=0)

    denominator = TP + FN
    denominator[denominator == 0] = 1  # 防止除零

    class_accuracy = TP / denominator

    try:
        import pandas as pd
        acc_df = pd.DataFrame({
            "Class": target_classes,
            "Accuracy": class_accuracy
        })
        print(acc_df)
    except Exception:
        print({"Class": target_classes, "Accuracy": class_accuracy})


def save_results(folder_path, acc,  precision, recall, f1, macro_ap, micro_ap, table_data):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 生成文件名，带时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(folder_path, f"results_{timestamp}.txt")

    with open(result_file, "w") as f:
        f.write("The training parameters are as follows\n")
        f.write(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
        f.write("\n\n***********************************************************************\n\n")
        f.write(f"Run Timestamp: {timestamp}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-averaged Precision: {precision:.4f}\n")
        f.write(f"Macro-averaged Recall: {recall:.4f}\n")
        f.write(f"Macro-averaged F1: {f1:.4f}\n")
        f.write(f"Macro-averaged AP: {macro_ap:.4f}\n")
        f.write(f"Micro-averaged AP: {micro_ap:.4f}\n")
        f.write("Metrics:\n")
    
    print(f"Results saved to {result_file}")
