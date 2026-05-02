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
    parser.add_argument('--data_root', type=str, default='../../data/CamVid/CamVid(11)', help='Dataset root path')
    parser.add_argument('--data_name', type=str, default='CamVid', help='Dataset class names')
    parser.add_argument('--model', type=str, default='segnet', help='Segmentation model')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint path')
    return parser.parse_args()

def train(args):
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {n_gpu}")

    # Dataloader
    train_loader, val_loader = get_dataloader(args.data_root, batch_size=args.batch_size)
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    print(f"Train samples: {train_dataset_size}, Val samples: {val_dataset_size}")

    # Model
    model = get_model(num_classes=args.num_classes)
    model.to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    # Resume
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint (epoch {start_epoch})")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'pixel_accuracy': [],
        'miou': []
    }

    print(f"🚀 Start training ({args.model})")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            images = images.to(device)
            masks = masks.to(device)


            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= train_dataset_size
        history['train_loss'].append(train_loss)

        # Validation
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

        history['pixel_accuracy'].append(scores['Pixel Accuracy'])
        history['miou'].append(scores['Mean Intersection over Union(mIoU)'])

        # Save best
        if scores['Mean Intersection over Union(mIoU)'] > best_miou:
            best_miou = scores['Mean Intersection over Union(mIoU)']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(args.checkpoint, f'{args.model}_best_{best_miou}.pth'))
            print(f"Saved best model! mIoU: {best_miou:.4f}")

        scheduler.step()

        print(f"🕒 Epoch time: {time.time() - t0:.2f}s\n")

    print("🎉 Training complete!")

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
