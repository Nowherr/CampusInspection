import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.Camvid_dataloader11_v3plus import get_dataloader
from model.deeplabv3plus import Deeplabv3plus
from metric import SegmentationMetric

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./CamVid')
    parser.add_argument('--data_name', type=str, default='CamVid')
    parser.add_argument('--model', type=str, default='deeplabv3plus')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.01, help='Head LR (backbone uses 0.1x)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint_deeplabv3plus')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignore_index', type=int, default=0)

    # ---- Loss / class-weights ----
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--weight_mode', type=str, default='sqrt_mfb',
                        choices=['mfb', 'sqrt_mfb', 'inv_log', 'none'],
                        help='Strategy for computing class weights')
    parser.add_argument('--w_min', type=float, default=0.5,
                        help='Lower clip for class weights')
    parser.add_argument('--w_max', type=float, default=2.5,
                        help='Upper clip for class weights')
    parser.add_argument('--label_smoothing', type=float, default=0.05)

    # ---- OHEM (default OFF for small dataset) ----
    parser.add_argument('--use_ohem', action='store_true', default=False,
                        help='Use Online Hard Example Mining CE')
    parser.add_argument('--ohem_thresh', type=float, default=0.7)
    parser.add_argument('--ohem_min_kept', type=int, default=100000)

    # ---- LR schedule ----
    parser.add_argument('--poly_power', type=float, default=0.9)
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='Linear warmup iterations from 0 to base_lr')
    parser.add_argument('--grad_clip', type=float, default=10.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# OHEM Cross Entropy Loss  (kept for optional later finetune)
# ---------------------------------------------------------------------------
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super().__init__()
        self.thresh_loss = float(-np.log(thresh))
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, target):
        pixel_losses = self.criterion(logits, target).view(-1)
        valid_mask = target.view(-1) != self.ignore_index
        valid_losses = pixel_losses[valid_mask]

        if valid_losses.numel() == 0:
            return pixel_losses.sum() * 0.0

        k = min(self.min_kept, valid_losses.numel())
        sorted_losses, _ = torch.sort(valid_losses, descending=True)
        topk_thresh = sorted_losses[k - 1].item()
        final_thresh = max(self.thresh_loss, topk_thresh)

        kept = valid_losses[valid_losses >= final_thresh]
        if kept.numel() == 0:
            return valid_losses.mean()
        return kept.mean()


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(train_loader, num_classes, ignore_index=0,
                          mode='sqrt_mfb', w_min=0.5, w_max=2.5):
    """
    mode:
      'mfb'      : median / freq                (aggressive, original)
      'sqrt_mfb' : sqrt(median / freq)          ← recommended for small dataset
      'inv_log'  : 1 / log(1.02 + freq)
      'none'     : all ones (no weighting)
    """
    print(f"📊 Computing class frequencies for weighted loss (mode={mode}) ...")
    class_pixel_count = torch.zeros(num_classes, dtype=torch.float64)
    for _, masks in tqdm(train_loader, desc='Counting pixels'):
        for c in range(num_classes):
            class_pixel_count[c] += (masks == c).sum().item()

    total = class_pixel_count.sum()
    freq = class_pixel_count / (total + 1e-12)
    freq_np = freq.numpy()
    present = freq_np > 0
    freq_safe = np.where(present, freq_np, 1.0)

    if mode == 'mfb':
        med = float(np.median(freq_np[present]))
        w = med / freq_safe
    elif mode == 'sqrt_mfb':
        med = float(np.median(freq_np[present]))
        w = np.sqrt(med / freq_safe)
    elif mode == 'inv_log':
        w = 1.0 / np.log(1.02 + freq_safe)
    elif mode == 'none':
        w = np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    # zero-out absent classes before clipping (so clip lower-bound doesn't revive them)
    w = np.where(present, w, 0.0)
    # clip only on the present classes
    w_clipped = np.clip(w, w_min, w_max)
    w_clipped = np.where(present, w_clipped, 0.0)

    if 0 <= ignore_index < num_classes:
        w_clipped[ignore_index] = 0.0

    weights = torch.tensor(w_clipped, dtype=torch.float32)
    print(f"Class pixel counts: {class_pixel_count.long().tolist()}")
    print(f"Class frequencies : {freq_np.round(5).tolist()}")
    print(f"Class weights     : {weights.numpy().round(3).tolist()}")
    return weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    os.makedirs(args.checkpoint, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {n_gpu}")

    # --- Data ---
    train_loader, val_loader = get_dataloader(args.data_root, batch_size=args.batch_size)
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    print(f"Train samples: {train_dataset_size}, Val samples: {val_dataset_size}")

    # --- Model ---
    model = Deeplabv3plus(backbone=args.backbone, num_classes=args.num_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model: DeepLabV3+ | backbone={args.backbone} | #params={n_params:.2f}M")

    # --- Loss ---
    class_weights = None
    if args.use_class_weights and args.weight_mode != 'none':
        class_weights = compute_class_weights(
            train_loader, args.num_classes,
            ignore_index=args.ignore_index,
            mode=args.weight_mode,
            w_min=args.w_min, w_max=args.w_max,
        ).to(device)

    if args.use_ohem:
        criterion = OhemCrossEntropy(
            ignore_index=args.ignore_index,
            thresh=args.ohem_thresh,
            min_kept=args.ohem_min_kept,
            weight=class_weights,
        )
        print(f"✅ Using OHEM CE (thresh={args.ohem_thresh}, min_kept={args.ohem_min_kept})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=args.ignore_index,
            label_smoothing=args.label_smoothing,
        )
        print(f"✅ Using weighted CE (label_smoothing={args.label_smoothing})")

    # --- Optimizer: backbone uses 0.1x LR ---
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    base_lrs = [args.lr * 0.1, args.lr]
    optimizer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': base_lrs[0]},
            {'params': head_params,     'lr': base_lrs[1]},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # --- Resume ---
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt['epoch']
        best_miou = ckpt.get('best_miou', 0.0)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️  optimizer state not loaded: {e}")
        print(f"Loaded checkpoint (epoch {start_epoch}, best_miou {best_miou:.4f})")

    history = {'train_loss': [], 'val_loss': [],
               'pixel_accuracy': [], 'miou_all': [], 'miou': []}

    total_iters = args.epochs * len(train_loader)
    cur_iter = start_epoch * len(train_loader)
    warmup_iters = max(0, int(args.warmup_iters))

    print(f"🚀 Start training ({args.model} / {args.backbone}) | "
          f"total iters: {total_iters} | warmup: {warmup_iters}")

    for epoch in range(start_epoch, args.epochs):
        # ---------------- Train ----------------
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, masks in tqdm(train_loader,
                                  desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            # ---- LR schedule: linear warmup + poly decay ----
            if cur_iter < warmup_iters:
                lr_scale = (cur_iter + 1) / float(warmup_iters)
            else:
                progress = (cur_iter - warmup_iters) / max(1, total_iters - warmup_iters)
                progress = min(1.0, max(0.0, progress))
                lr_scale = (1.0 - progress) ** args.poly_power
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = base_lrs[i] * lr_scale

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            cur_iter += 1

        train_loss /= train_dataset_size
        history['train_loss'].append(train_loss)
        cur_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} "
              f"| LR(bb/head): {cur_lrs[0]:.6f} / {cur_lrs[1]:.6f}")

        # ---------------- Validate ----------------
        model.eval()
        val_loss = 0.0
        evaluator = SegmentationMetric(args.num_classes)
        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                      desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                evaluator.addBatch(preds, masks.cpu().numpy())

        val_loss /= val_dataset_size
        history['val_loss'].append(val_loss)

        scores = evaluator.get_scores()
        print(f"\n📈 Validation Epoch {epoch+1}:")
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {np.round(v, 3)}")
            else:
                print(f"{k}: {v:.4f}")

        # ---- Fair mIoU: exclude ignore_index (void) ----
        iou_per_class = scores.get('Intersection over Union', None)
        miou_all = scores['Mean Intersection over Union(mIoU)']
        if isinstance(iou_per_class, np.ndarray):
            mask_valid = np.ones(args.num_classes, dtype=bool)
            if 0 <= args.ignore_index < args.num_classes:
                mask_valid[args.ignore_index] = False
            miou = float(np.nanmean(iou_per_class[mask_valid]))
            print(f"🎯 mIoU (exclude class {args.ignore_index}): {miou:.4f}   "
                  f"| mIoU (all): {miou_all:.4f}")
        else:
            miou = miou_all

        history['pixel_accuracy'].append(scores['Pixel Accuracy'])
        history['miou_all'].append(miou_all)
        history['miou'].append(miou)

        # ---- Save best ----
        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(
                args.checkpoint, f'{args.model}_{args.backbone}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'backbone': args.backbone,
            }, ckpt_path)
            print(f"💾 Saved best model ({ckpt_path}) | mIoU: {best_miou:.4f}")

        print(f"🕒 Epoch time: {time.time() - t0:.2f}s\n")

    print(f"🎉 Training complete! Best mIoU (exclude void): {best_miou:.4f}")


if __name__ == '__main__':
    args = parse_arguments()
    train(args)