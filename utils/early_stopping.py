import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Automatically saves best model checkpoint (model + optimizer + epoch).
    """

    def __init__(self, patience=7, mode="min", delta=0.0, save_path=None):
        """
        Args:
            patience (int): how many epochs to wait before stopping after no improvement
            mode (str): 'min' for loss/MAE, 'max' for accuracy/F1
            delta (float): minimum change to qualify as improvement
            save_path (str): where to save the best model
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, current_score, model, optimizer=None, epoch=None):
        """Check whether the metric improved."""
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self._save_checkpoint(model, optimizer, epoch)
            print(f"Initial model saved (val={current_score:.4f})")
            return

        improved = False
        if self.mode == "min":
            improved = current_score < self.best_score - self.delta
        elif self.mode == "max":
            improved = current_score > self.best_score + self.delta

        if improved:
            print(f"✅ Metric improved: {self.best_score:.4f} → {current_score:.4f}")
            self.best_score = current_score
            self.best_epoch = epoch
            self._save_checkpoint(model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {self.best_epoch}")
                self.early_stop = True

    def _save_checkpoint(self, model, optimizer=None, epoch=None):
        """Save model, optimizer, and epoch state."""
        if self.save_path:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "epoch": epoch if epoch is not None else 0
            }
            torch.save(checkpoint, self.save_path)
            print(f"Saved new best model to {self.save_path}")
