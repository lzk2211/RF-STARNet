import os
import random
import torch
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, args=None):
        self.args = args
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample_file = self.file_paths[idx]
        sample_labels = self.labels[idx]

        parts = sample_file.split('/')
        floader = parts[-3]
        snr = 'pure' if floader == 'data_set_5e6' else floader.rsplit("5e6_", 1)[1]

        img_pil = Image.open(sample_file).convert('L')

        if self.transform is not None:
            image = self.transform(img_pil)
        else:
            image = transforms.ToTensor()(img_pil)  # [1,H,W], float32 [0,1]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image))

        return image, sample_labels, snr

def data_set_split(nw, args):

    RANDOM_SEED = args.seed
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"SEED set to: {RANDOM_SEED}")

    label_map = {}

    data_paths = ['-20', '-15', '-10', '-5', '0', '5']
    data_paths = ['_' + path + 'dB/' for path in data_paths]
    data_paths.append('/')
    data_abs_paths = [args.root_path + path for path in data_paths]

    num_classes = 10
    all_classes =   ["T0000", "T0001", "T0010", "T0011",
                    "T0100", "T0101", "T0110", "T0111",
                    "T1000", "T1001", "T1010", "T1011",
                    "T1100", "T1101", "T1110", "T1111",
                    "T10000", "T10001", "T10010", "T10011",
                    "T10100", "T10101", "T10110", "T10111", "T11000"]

    target_classes = all_classes[:num_classes]
    label_map = {cls: i for i, cls in enumerate(target_classes)}

    train_dataset_paths = []
    train_labels_str = []

    val_dataset_paths = []
    val_labels_str = []

    test_dataset_paths = []
    test_labels_str = []

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2 # TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0

    for data_path in data_abs_paths:
        for target_class in target_classes:
            label_folder_path = os.path.join(data_path, target_class)
            
            if os.path.isdir(label_folder_path):
                all_files_in_folder = os.listdir(label_folder_path)
                all_file_paths = [os.path.join(label_folder_path, f) 
                                for f in all_files_in_folder if os.path.isfile(os.path.join(label_folder_path, f))]

                if not all_file_paths:
                    print(f"  - 警告: 目录 {label_folder_path} 为空, 已跳过。")
                    continue

                total_samples = len(all_file_paths)

                ###########################   Data Sampling Code   ###########################
                total_samples = total_samples * args.ratio # Take proportional samples
                ###########################   Data Sampling Code   ###########################

                indices = list(range(int(total_samples)))
                
                random.shuffle(indices) 
                                
                train_size = int(total_samples * TRAIN_RATIO) 
                val_size = int(total_samples * VAL_RATIO)
                test_size = total_samples - train_size - val_size 
                                
                train_indices = indices[:train_size]
                val_indices = indices[train_size : train_size + val_size]
                test_indices = indices[train_size + val_size:]
                
                train_dataset_paths.extend([all_file_paths[i] for i in train_indices])
                train_labels_str.extend([target_class] * len(train_indices))
                
                val_dataset_paths.extend([all_file_paths[i] for i in val_indices])
                val_labels_str.extend([target_class] * len(val_indices))

                test_dataset_paths.extend([all_file_paths[i] for i in test_indices])
                test_labels_str.extend([target_class] * len(test_indices))

    print("...Data partitioning completed.")

    train_labels = [label_map[cls_name] for cls_name in train_labels_str]
    val_labels = [label_map[cls_name] for cls_name in val_labels_str]
    test_labels = [label_map[cls_name] for cls_name in test_labels_str]

    print(f"\nOverview of Dataset Partitioning:")
    print(f"  - Number of training samples: {len(train_dataset_paths)}")
    print(f"  - Number of training labels: {len(train_labels)}")
    print(f"  - Number of validation samples: {len(val_dataset_paths)}")
    print(f"  - Number of validation labels: {len(val_labels)}")
    print(f"  - Number of test samples: {len(test_dataset_paths)}")
    print(f"  - Number of test labels: {len(test_labels)}")
    print(f"  - Total number of samples: {len(train_dataset_paths) + len(val_dataset_paths) + len(test_dataset_paths)}")

    resize = 512
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),
        ])
    }
    train_dataset = CustomDataset(train_dataset_paths, train_labels, transform=data_transform["train"], args=args)
    validate_dataset = CustomDataset(val_dataset_paths, val_labels, transform=data_transform["val"], args=args)
    test_dataset = CustomDataset(test_dataset_paths, test_labels, transform=data_transform["test"], args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    test_num = len(test_dataset)
    
    print(f"Training samples: {train_num}")
    print(f"Validation samples: {val_num}")
    print(f"Testing samples: {test_num}")

    return train_loader, validate_loader, test_loader, target_classes
