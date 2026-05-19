# inference/segmenter.py
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# 把项目根目录加进 sys.path，确保能 import model.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.deeplabv3plus import Deeplabv3plus


# ============ 关键：与训练完全一致的 12 类（void=0） ============
CLASS_NAMES = [
    'void', 'Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
]

CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
# 也就是 -> {'void':0,'Sky':1,'Building':2,'Pole':3,'Road':4,'Sidewalk':5,
#           'Tree':6,'SignSymbol':7,'Fence':8,'Car':9,'Pedestrian':10,'Bicyclist':11}

# 与 test_deeplabv3plus.py 默认 colormap 保持一致
PALETTE = np.array([
    [0,   0,   0  ],   # 0  void
    [128, 128, 128],   # 1  Sky
    [128, 0,   0  ],   # 2  Building
    [192, 192, 128],   # 3  Pole
    [128, 64,  128],   # 4  Road
    [0,   0,   192],   # 5  Sidewalk
    [128, 128, 0  ],   # 6  Tree
    [192, 128, 128],   # 7  SignSymbol
    [64,  64,  128],   # 8  Fence
    [64,  0,   128],   # 9  Car
    [64,  64,  0  ],   # 10 Pedestrian
    [0,   128, 192],   # 11 Bicyclist
], dtype=np.uint8)


class CampusSegmenter:
    def __init__(self,
                 checkpoint_path='./checkpoint_deeplabv3plus/deeplabv3plus_resnet50_best.pth',
                 backbone='resnet50',
                 num_classes=12,
                 ignore_index=0,
                 device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # 1. 构建模型
        self.model = Deeplabv3plus(backbone=backbone, num_classes=num_classes)

        # 2. 加载 checkpoint（完全沿用你 test 脚本的逻辑）
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            self.meta = {
                'epoch': ckpt.get('epoch', '?'),
                'best_miou': ckpt.get('best_miou', None),
                'backbone': ckpt.get('backbone', backbone),
            }
        else:
            state_dict = ckpt
            self.meta = {'epoch': '?', 'best_miou': None, 'backbone': backbone}

        # 去掉 DataParallel 前缀
        new_sd = {(k[7:] if k.startswith('module.') else k): v
                  for k, v in state_dict.items()}
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device).eval()

        # 3. 预处理（与训练/测试一致）
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_input):
        """
        输入：可以是 PIL.Image / numpy RGB / 文件路径
        输出：mask (H, W) np.uint8，每个像素是 0~11 的类别 id
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        W, H = pil_img.size
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        out = self.model(x)
        # 保险：上采样回原图大小
        if out.shape[-2:] != (H, W):
            out = torch.nn.functional.interpolate(
                out, size=(H, W), mode='bilinear', align_corners=False
            )

        # 屏蔽 void
        if 0 <= self.ignore_index < self.num_classes:
            out[:, self.ignore_index, :, :] = -1e4

        mask = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask

    def colorize(self, mask):
        """把 id mask 上色 (RGB)"""
        return PALETTE[mask]

    def get_class_ratio(self, mask):
        """各类像素占比"""
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        return {CLASS_NAMES[int(u)]: float(c) / total
                for u, c in zip(unique, counts)}