import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# ---------------- Albumentations version compat ----------------
# In albumentations >= 1.4.0, PadIfNeeded uses `fill` / `fill_mask`.
# In older versions it uses `value` / `mask_value`.
try:
    from packaging import version as _v
    _ALB_NEW = _v.parse(A.__version__) >= _v.parse("1.4.0")
except Exception:
    # Fallback: simple string compare on major.minor
    try:
        _major, _minor = A.__version__.split('.')[:2]
        _ALB_NEW = (int(_major), int(_minor)) >= (1, 4)
    except Exception:
        _ALB_NEW = False


def _pad_if_needed(crop_size):
    """Version-agnostic PadIfNeeded with zero padding for both image & mask."""
    kwargs = dict(min_height=crop_size, min_width=crop_size, border_mode=0)
    if _ALB_NEW:
        kwargs.update(fill=0, fill_mask=0)
    else:
        kwargs.update(value=0, mask_value=0)
    return A.PadIfNeeded(**kwargs)


# ---------------- CamVid 12 classes ----------------
Cam_CLASSES = ["Unlabelled", "Sky", "Building", "Pole",
               "Road", "Sidewalk", "Tree", "SignSymbol",
               "Fence", "Car", "Pedestrian", "Bicyclist"]

Cam_COLORMAP = [
    [0, 0, 0],   [128, 128, 128], [128, 0, 0],   [192, 192, 128],
    [128, 64, 128], [0, 0, 192],  [128, 128, 0], [192, 128, 128],
    [64, 64, 128], [64, 0, 128],  [64, 64, 0],   [0, 128, 192]
]

# ---------------- ImageNet stats (match pretrained backbone) ----------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- Crop / image sizes ----------------
# CamVid images are typically 480x360 (sometimes 960x720).
# We train on 480x480 random crops, val on full 480x720 resize.
TRAIN_CROP = 480
VAL_H, VAL_W = 480, 640   # keep close to native aspect ratio

# ---------------- Fast RGB -> class-id lookup ----------------
# Encode RGB triplet into a single int and build a dict lookup.
_COLOR_LUT = {}
for _idx, _c in enumerate(Cam_COLORMAP):
    _key = (int(_c[0]) << 16) | (int(_c[1]) << 8) | int(_c[2])
    _COLOR_LUT[_key] = _idx


def mask_to_class(mask_rgb):
    """mask_rgb: (H, W, 3) uint8  ->  (H, W) int64 class map (unknown -> 0)."""
    m = mask_rgb.astype(np.int32)
    keys = (m[..., 0] << 16) | (m[..., 1] << 8) | m[..., 2]
    out = np.zeros(keys.shape, dtype=np.int64)  # default: 0 (Unlabelled)
    for k, v in _COLOR_LUT.items():
        if v == 0:
            continue
        out[keys == k] = v
    return out


# ---------------- Transforms ----------------
def build_train_transform(crop_size=TRAIN_CROP):
    return A.Compose([
        # random scaling 0.5x ~ 2.0x (standard DeepLab recipe)
        A.SmallestMaxSize(max_size=int(crop_size * 1.2), interpolation=1),
        A.RandomScale(scale_limit=(-0.5, 1.0), interpolation=1, p=1.0),
        _pad_if_needed(crop_size),
        A.RandomCrop(height=crop_size, width=crop_size),

        A.HorizontalFlip(p=0.5),

        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_val_transform(h=VAL_H, w=VAL_W):
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# ---------------- Dataset ----------------
class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        assert len(self.images) == len(self.labels), \
            f"Images ({len(self.images)}) vs labels ({len(self.labels)}) mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        lbl_path = os.path.join(self.label_dir, self.labels[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        label_rgb = np.array(Image.open(lbl_path).convert("RGB"))
        mask = mask_to_class(label_rgb)  # (H, W) int64

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].long()


# ---------------- Dataloader factory ----------------
def get_dataloader(data_path, batch_size=8, num_workers=4,
                   crop_size=TRAIN_CROP, val_size=(VAL_H, VAL_W)):
    train_dir      = os.path.join(data_path, 'train')
    val_dir        = os.path.join(data_path, 'val')
    trainlabel_dir = os.path.join(data_path, 'train_labels')
    vallabel_dir   = os.path.join(data_path, 'val_labels')

    train_tf = build_train_transform(crop_size=crop_size)
    val_tf   = build_val_transform(h=val_size[0], w=val_size[1])

    train_dataset = CamVidDataset(train_dir, trainlabel_dir, train_tf)
    val_dataset   = CamVidDataset(val_dir,   vallabel_dir,   val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ---------------- Quick sanity check ----------------
if __name__ == "__main__":
    tr, va = get_dataloader('./CamVid', batch_size=2, num_workers=0)
    print(f"train batches: {len(tr)}, val batches: {len(va)}")
    for img, mask in tr:
        print("train img:", img.shape, img.dtype, "| mask:", mask.shape, mask.dtype,
              "| mask unique:", torch.unique(mask).tolist())
        break
    for img, mask in va:
        print("val   img:", img.shape, img.dtype, "| mask:", mask.shape, mask.dtype)
        break