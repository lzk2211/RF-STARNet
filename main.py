import os
import random
import torch
from tqdm import tqdm
import argparse
import numpy as np
import json
from dataset.ZJU_dataset_pre_multiSNR import data_set_split
from fun import folder_settings, switch_model

def set_seed(seed: int):
    """Set random seeds for reproducibility before model initialization."""
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN and PyTorch ops deterministic where possible
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            # Full determinism (may impact performance; warn_only avoids hard errors on unsupported ops)
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        except Exception:
            pass
    except Exception as e:
        print(f"[seed] Failed to set full determinism: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train a localization model")
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--resume', type=bool, default=False, help='train from checkpoint')
    parser.add_argument('--ratio', type=float, default=1, help='ratio')

    parser.add_argument('--mode', type=str, default='Test', choices=['Train', 'Test'], help='switch mode')
    parser.add_argument('--exp_name', type=str, default='F_T_small_mulitloss_1122', help='the name of the experiment')

    parser.add_argument('--class_num', type=int, default=10, help='class number')
    parser.add_argument('--model', type=str, default='F_T_small_mulitloss_1122', help='switch model')
    parser.add_argument('--root_path', type=str, default='/home/kk/D00/data_set_5e6', help='dataset path')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/testing')

    args = parser.parse_args()
    print(args)


    folder_path = './final_results/' + str(args.seed) + '/'  + args.exp_name
    if not os.path.exists(folder_path):# if the folder does not exist, than create it
        os.makedirs(folder_path)

    if args.mode == 'Train':
        if not isinstance(args, dict):
            config_dict = vars(args)
        else:
            config_dict = args
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    elif args.mode == 'Test':
        config_path = os.path.join(folder_path, "config.json") 
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        keys_to_skip_update = ['resume', 'TSNE', 'mode', 'config', 'device']
        for key, value in loaded_config.items():
            if hasattr(args, key) and (key not in keys_to_skip_update):
                setattr(args, key, value)

    device, nw, table_data = folder_settings(args, folder_path)

    # Update the model to fit your input size and number of classes
    model = switch_model(args.model, device, args.class_num)

    if 'data_set' in args.root_path:
        train_loader, validate_loader, test_loader, target_classes = data_set_split(nw, args)
    print(target_classes)
    
    if args.mode == 'Train':
        if 'mulitloss' in args.model:
            from train_val_test_multiSNR_mulitloss import train, test
        else:
            from train_val_test_multiSNR import train, test
        train(args, folder_path, model, device, train_loader, validate_loader)
        test(args, folder_path, model, device, test_loader, target_classes)

    if args.mode == 'Test':
        if 'mulitloss' in args.model:
            from train_val_test_multiSNR_mulitloss import train, test
        else:
            from train_val_test_multiSNR import train, test
        test(args, folder_path, model, device, test_loader, target_classes)

if __name__ == '__main__':
    main()
