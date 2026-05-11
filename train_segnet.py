import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from datasets.CamVid_dataloader11 import get_dataloader
from model import get_model
from metric import SegmentationMetric

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./CamVid', help='Dataset root path')
    parser.add_argument('--data_name', type=str, default='CamVid', help='Dataset class names')
    parser.add_argument('--model', type=str, default='segnet', help='Segmentation model')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    parser.add_argument('--ignore_index', type=int, default=0, help='Class index to ignore in loss')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use Median Frequency Balancing class weights')
    return parser.parse_args()


def compute_class_weights(train_loader, num_classes, ignore_index=0):
    """
    Median Frequency Balancing (Eigen & Fergus 2015).
    weight[c] = median(freq) / freq[c]
    """
    print("📊 Computing class frequencies for weighted loss ...")
    class_pixel_count = torch.zeros(num_classes, dtype=torch.float64)

    for _, masks in tqdm(train_loader, desc='Counting pixels'):
        # masks: (B, H, W) long tensor
        for c in range(num_classes):
            class_pixel_count[c] += (masks == c).sum().item()

    total = class_pixel_count.sum()
    freq = class_pixel_count / (total + 1e-12)

    # Use median frequency only over classes that actually appear
    present = freq > 0
    if present.sum() == 0:
        raise RuntimeError("No classes found in training set!")
    median_freq = torch.median(freq[present])

    weights = torch.zeros(num_classes, dtype=torch.float32)
    weights[present] = (median_freq / freq[present]).float()

    # Zero out ignore_index so it doesn't perturb anything (loss will skip it anyway)
    if 0 <= ignore_index < num_classes:
        weights[ignore_index] = 0.0

    # Cap the maximum weight to avoid instability when a class is extremely rare
    max_cap = 10.0
    weights = torch.clamp(weights, max=max_cap)

    print(f"Class pixel counts: {class_pixel_count.long().tolist()}")
    print(f"Class frequencies : {freq.numpy().round(5).tolist()}")
    print(f"Class weights     : {weights.numpy().round(3).tolist()}")
    return weights


def train(args):
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {n_gpu}")

    # --- Dataloader -----------------------------------------------------------
    train_loader, val_loader = get_dataloader(args.data_root, batch_size=args.batch_size)
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    print(f"Train samples: {train_dataset_size}, Val samples: {val_dataset_size}")

    # --- Model (with VGG16 pretrained encoder) --------------------------------
    model = get_model(num_classes=args.num_classes)
    model.to(device)

    # --- Loss: Weighted CE ----------------------------------------------------
    if args.use_class_weights:
        class_weights = compute_class_weights(train_loader, args.num_classes,
                                              ignore_index=args.ignore_index)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_index)
        print("✅ Using class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
        print("✅ Using plain CrossEntropyLoss.")

    # --- Optimizer ------------------------------------------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # --- Scheduler: ReduceLROnPlateau on val mIoU -----------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # --- Resume ---------------------------------------------------------------
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint.get('best_miou', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️  Skipped loading scheduler state: {e}")
        print(f"Loaded checkpoint (epoch {start_epoch}, best_miou {best_miou:.4f})")

    # --- History --------------------------------------------------------------
    history = {
        'train_loss': [],
        'val_loss': [],
        'pixel_accuracy': [],
        'miou': []
    }

    print(f"🚀 Start training ({args.model})")
    for epoch in range(start_epoch, args.epochs):
        # -------- Train --------
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= train_dataset_size
        history['train_loss'].append(train_loss)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # -------- Validate --------
        model.eval()
        val_loss = 0.0
        evaluator = SegmentationMetric(args.num_classes)
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]'):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                predictions = torch.argmax(outputs, dim=1)
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()

                evaluator.addBatch(predictions, masks)

        val_loss /= val_dataset_size
        history['val_loss'].append(val_loss)

        scores = evaluator.get_scores()
        print(f"\n📈 Validation Epoch {epoch+1}:")
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {np.round(v, 3)}")
            else:
                print(f"{k}: {v:.4f}")

        miou = scores['Mean Intersection over Union(mIoU)']
        history['pixel_accuracy'].append(scores['Pixel Accuracy'])
        history['miou'].append(miou)

        # -------- Save best -------- (fixed filename, overwrites)
        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(args.checkpoint, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, ckpt_path)
            print(f"💾 Saved best model ({ckpt_path}) | mIoU: {best_miou:.4f}")

        # -------- Scheduler step on val mIoU --------
        scheduler.step(miou)

        print(f"🕒 Epoch time: {time.time() - t0:.2f}s\n")

    print(f"🎉 Training complete! Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    args = parse_arguments()
    train(args)