import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice loss with ignore_index handling.
    AMP-safe: forces FP32 internally to avoid FP16 overflow/underflow.
    """
    def __init__(self, num_classes, smooth=1.0, ignore_index=0, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        # ✅ FIX 1: Force FP32 computation to prevent AMP/FP16 numerical issues
        logits = logits.float()

        # ✅ FIX 2: Clamp logits to a safe range before softmax
        # Prevents inf when model outputs extreme values during early training
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        probs = F.softmax(logits, dim=1)

        valid_mask = (targets != self.ignore_index)              # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                            # avoid index error

        onehot = F.one_hot(targets_safe.long(), self.num_classes)
        onehot = onehot.permute(0, 3, 1, 2).float()              # [B, C, H, W]

        vm = valid_mask.unsqueeze(1).float()                     # [B, 1, H, W]
        onehot = onehot * vm
        probs = probs * vm

        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = (probs + onehot).sum(dims)

        # ✅ FIX 3: Add eps for extra safety on top of smooth
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)

        # Exclude ignore_index class from mean
        if 0 <= self.ignore_index < self.num_classes:
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=dice.device)
            mask[self.ignore_index] = False
            dice = dice[mask]

        loss = 1.0 - dice.mean()

        # ✅ FIX 4: Final NaN/Inf guard (defensive)
        if not torch.isfinite(loss):
            print(f"⚠️ DiceLoss became non-finite: {loss.item()}, returning 0")
            loss = torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)

        return loss


class CEDiceLoss(nn.Module):
    """
    Wrapper that ADDS a Dice term to your existing CE / OHEM-CE.
    Keeps class weights, label_smoothing, OHEM intact.
    Total: ce_loss + dice_weight * dice_loss

    AMP-safe: each component is checked for NaN/Inf separately.
    """
    def __init__(self, ce_criterion, num_classes, ignore_index=0, dice_weight=0.5):
        super().__init__()
        self.ce = ce_criterion
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self._nan_count = 0

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        # ✅ Diagnostic: report which component is bad on first occurrence
        if not torch.isfinite(ce_loss):
            self._nan_count += 1
            if self._nan_count <= 3:
                print(f"⚠️ CE loss is non-finite: {ce_loss.item():.4f} "
                      f"| logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            ce_loss = torch.zeros_like(ce_loss, requires_grad=True)

        if not torch.isfinite(dice_loss):
            self._nan_count += 1
            if self._nan_count <= 3:
                print(f"⚠️ Dice loss is non-finite: {dice_loss.item():.4f}")
            dice_loss = torch.zeros_like(dice_loss, requires_grad=True)

        return ce_loss + self.dice_weight * dice_loss