import os
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
# 11类
Cam_CLASSES = [ "Unlabelled","Sky","Building","Pole",
                "Road","Sidewalk", "Tree","SignSymbol",
                "Fence","Car","Pedestrian","Bicyclist"]

# 用于做可视化
Cam_COLORMAP = [
    [0, 0, 0],[128, 128, 128],[128, 0, 0],[192, 192, 128],
    [128, 64, 128],[0, 0, 192],[128, 128, 0],[192, 128, 128],
    [64, 64, 128],[64, 0, 128],[64, 64, 0],[0, 128, 192]
]


# 转换RGB mask为类别id的函数
def mask_to_class(mask):
    mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for idx, color in enumerate(Cam_COLORMAP):
        color = np.array(color)
        # 每个像素和当前颜色匹配
        matches = np.all(mask == color, axis=-1)
        mask_class[matches] = idx
    return mask_class


class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])

        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

        assert len(self.images) == len(self.labels), "Images and labels count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        label_rgb = np.array(Image.open(label_path).convert("RGB"))

        # RGB转类别索引
        mask = mask_to_class(label_rgb)
        #mask = torch.from_numpy(np.array(mask)).long()

        # Albumentations 需要 (H, W, 3) 和 (H, W)
        transformed = self.transform(image=image, mask=mask)

        return transformed['image'], transformed['mask'].long()


def get_dataloader(data_path, batch_size=4, num_workers=4):
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    trainlabel_dir = os.path.join(data_path, 'train_labels')
    vallabel_dir = os.path.join(data_path, 'val_labels')
    train_dataset = CamVidDataset(train_dir, trainlabel_dir)
    val_dataset = CamVidDataset(val_dir, vallabel_dir)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader
