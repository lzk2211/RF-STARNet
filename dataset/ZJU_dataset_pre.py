import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


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

        img_pil = Image.open(sample_file).convert('L')

        if self.transform is not None:
            image = self.transform(img_pil)
        else:
            image = transforms.ToTensor()(img_pil)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image))

        if self.args.Dim == 'f':
            image = torch.squeeze(torch.sum(image, dim=2, keepdim=True)).unsqueeze(0) # We output a 1D vector of frequency sequence
        elif self.args.Dim == 't':
            image = torch.squeeze(torch.sum(image, dim=1, keepdim=True)).unsqueeze(0) # We output a 1D vector of time sequence

        return image, sample_labels

def data_set_split(nw, args):
    label_map = {}

    num_classes = args.class_num
    all_classes =   ["T0000", "T0001", "T0010", "T0011",
                    "T0100", "T0101", "T0110", "T0111",
                    "T1000", "T1001", "T1010", "T1011",
                    "T1100", "T1101", "T1110", "T1111",
                    "T10000", "T10001", "T10010", "T10011",
                    "T10100", "T10101", "T10110", "T10111", "T11000"]

    target_classes = all_classes[:num_classes]
    label_map = {cls: i for i, cls in enumerate(target_classes)}
    
    if not hasattr(args, 'image_path'):
        raise ValueError("args must include the image_path parameter pointing to the dataset's path.")
        
    print("The current dataset is single-label. Use the single-label dataset.")
    test_dataset_paths = []
    test_labels_paths = []
    for target_class in target_classes:
        label_folder_path = os.path.join(args.image_path, target_class)
        if os.path.isdir(label_folder_path):
            for file_name in os.listdir(label_folder_path):
                test_dataset_paths.append(os.path.join(label_folder_path, file_name))
                test_labels_paths.append(target_class)

    test_labels_paths = [label_map[label] for label in test_labels_paths]
    test_labels_paths = torch.tensor(test_labels_paths)
    test_labels_paths = F.one_hot(test_labels_paths, num_classes=num_classes).float()

    resize = 512
    # 创建数据变换
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

    
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    total_samples = len(test_dataset_paths)
    train_size = int(total_samples * train_ratio)
    test_size = int(total_samples * val_ratio)
    val_size = int(total_samples * test_ratio)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    val_indices = indices[:val_size]
    test_indices = indices[val_size:test_size + val_size]
    train_indices = indices[test_size + val_size:train_size + test_size + val_size]
    
    train_files = [test_dataset_paths[i] for i in train_indices]
    train_labels = [test_labels_paths[i] for i in train_indices]

    val_files = [test_dataset_paths[i] for i in val_indices]
    val_labels = [test_labels_paths[i] for i in val_indices]
    
    test_files = [test_dataset_paths[i] for i in test_indices]
    test_labels = [test_labels_paths[i] for i in test_indices]

    train_dataset = CustomDataset(train_files, train_labels, transform=data_transform["train"], args=args)
    validate_dataset = CustomDataset(val_files, val_labels, transform=data_transform["val"], args=args)
    test_dataset = CustomDataset(test_files, test_labels, transform=data_transform["test"], args=args)

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
