# test_deeplabv3plus.py
import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from model.deeplabv3plus import Deeplabv3plus

# 尝试从 v3plus dataloader 拿 colormap，拿不到就用内置的
try:
    from datasets.Camvid_dataloader11_v3plus import Cam_COLORMAP
except Exception:
    try:
        from datasets.CamVid_dataloader11 import Cam_COLORMAP
    except Exception:
        # CamVid 12 类（含 void）默认调色板，顺序需与训练标签一致
        Cam_COLORMAP = [
            [0,   0,   0  ],   # 0  void / unlabelled
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
        ]

CLASS_NAMES = [
    'void', 'Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./datasets/test',
                        help='Input image file or folder')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint_deeplabv3plus/deeplabv3plus_resnet50_best.pth',
                        help='Checkpoint path')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--save_dir', type=str, default='./predictions_deeplabv3plus')
    parser.add_argument('--input_size', type=int, nargs=2, default=None,
                        help='Optional resize (H W). Default: keep original size.')
    parser.add_argument('--overlay', action='store_true', default=True,
                        help='Save overlay (raw image + predicted mask)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Overlay alpha for predicted mask')
    parser.add_argument('--ignore_index', type=int, default=0,
                        help='Class index to ignore (void). Set -1 to disable.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def build_transform(input_size=None):
    ops = []
    if input_size is not None:
        ops.append(T.Resize(tuple(input_size), interpolation=T.InterpolationMode.BILINEAR))
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)   # [1, 3, H, W]
    return tensor, image


def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(Cam_COLORMAP):
        color_mask[mask == label] = color
    return color_mask


def save_mask(mask, save_path):
    Image.fromarray(mask_to_color(mask)).save(save_path)


def overlay_mask_on_image(raw_image, mask, alpha=0.6):
    mask_color = mask_to_color(mask)
    mask_pil = Image.fromarray(mask_color).resize(raw_image.size, resample=Image.NEAREST)
    return Image.blend(raw_image.convert('RGB'), mask_pil, alpha=alpha)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args, device):
    model = Deeplabv3plus(backbone=args.backbone, num_classes=args.num_classes)

    # ✅ 优先使用 weights_only=True（更安全，且消除 FutureWarning）
    #    若 checkpoint 含有非安全对象（旧版 numpy 标量等），自动回退
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception as e:
        print(f"⚠️  weights_only=True 加载失败，回退到 weights_only=False：{e}")
        print("   （仅在你完全信任该 checkpoint 来源时才使用！）")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 兼容多种保存格式
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        ep   = ckpt.get('epoch', '?')
        miou = ckpt.get('best_miou', None)
        bb   = ckpt.get('backbone', args.backbone)
        msg = f"Loaded checkpoint: epoch={ep}, backbone={bb}"
        if miou is not None:
            try:
                msg += f", best_miou={float(miou):.4f}"
            except Exception:
                msg += f", best_miou={miou}"
        print(f"✅ {msg}")
    else:
        state_dict = ckpt
        print("✅ Loaded raw state_dict checkpoint")

    # 去掉 DataParallel 的 'module.' 前缀
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith('module.') else k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {len(missing)} (showing 5): {missing[:5]}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args, device)
    transform = build_transform(args.input_size)

    os.makedirs(args.save_dir, exist_ok=True)

    # 收集图片
    if os.path.isdir(args.image_dir):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_list = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(exts)
        ])
    elif os.path.isfile(args.image_dir):
        image_list = [args.image_dir]
    else:
        raise FileNotFoundError(f"Path not found: {args.image_dir}")

    print(f"🔎 Found {len(image_list)} image(s) to predict.")

    for img_path in tqdm(image_list, desc='Predicting'):
        img_tensor, raw_img = load_image(img_path, transform)
        img_tensor = img_tensor.to(device, non_blocking=True)

        output = model(img_tensor)                       # [1, C, h, w]
        # 保险起见上采样回原图尺寸（如果模型输出不是原尺寸的话）
        if output.shape[-2:] != (raw_img.size[1], raw_img.size[0]):
            output = torch.nn.functional.interpolate(
                output, size=(raw_img.size[1], raw_img.size[0]),
                mode='bilinear', align_corners=False
            )

        # 可选：把 ignore_index (void) 的 logit 压低，避免预测成 void
        if 0 <= args.ignore_index < args.num_classes:
            output[:, args.ignore_index, :, :] = -1e4

        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_save_path = os.path.join(args.save_dir, f"{base_name}_mask.png")
        save_mask(pred, mask_save_path)

        if args.overlay:
            overlay_img = overlay_mask_on_image(raw_img, pred, alpha=args.alpha)
            overlay_img.save(os.path.join(args.save_dir, f"{base_name}_overlay.png"))

        # 简单统计每张图各类占比（debug 友好）
        unique, counts = np.unique(pred, return_counts=True)
        ratio = {CLASS_NAMES[u] if u < len(CLASS_NAMES) else str(u):
                 round(c / pred.size, 3) for u, c in zip(unique, counts)}
        tqdm.write(f"  {base_name} -> {ratio}")

    print(f"🎉 Done! Predictions saved to: {args.save_dir}")


if __name__ == '__main__':
    args = parse_arguments()
    predict(args)