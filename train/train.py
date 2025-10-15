"""
Training loop with Automatic Mixed Precision (AMP).

Features:
- Combined policy (CE) + value (MSE) loss
- AMP with bf16/fp16
- Cosine LR schedule
- Checkpointing
- Rich progress logging
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from model import create_model_main, create_model_mini
from train.dataset import ChessPositionDataset


class ChessLoss(nn.Module):
    """
    Combined policy + value loss.

    Loss = CE(policy) + lambda * MSE(value)
    """

    def __init__(self, value_weight: float = 0.25):
        super().__init__()
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            policy_logits: (B, 8, 8, 73) predicted logits
            value_pred: (B, 1) predicted values
            policy_target: (B, 8, 8, 73) target distribution
            value_target: (B,) target values

        Returns:
            (total_loss, policy_loss, value_loss)
        """
        # Policy loss (cross-entropy)
        # Flatten spatial and action dimensions
        B = policy_logits.shape[0]
        policy_logits_flat = policy_logits.reshape(B, -1)  # (B, 8*8*73)
        policy_target_flat = policy_target.reshape(B, -1)  # (B, 8*8*73)

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(policy_logits_flat, dim=1)
        policy_loss = -(policy_target_flat * log_probs).sum(dim=1).mean()

        # Value loss (MSE)
        value_pred = value_pred.squeeze(1)  # (B,)
        value_loss = F.mse_loss(value_pred, value_target)

        # Combined loss
        total_loss = policy_loss + self.value_weight * value_loss

        return total_loss, policy_loss, value_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ChessLoss,
    device: str,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary with metrics
    """
    model.train()
    scaler = GradScaler() if use_amp else None

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    for batch in dataloader:
        board_tensor = batch["board_tensor"].to(device)
        policy_target = batch["policy_target"].to(device)
        value_target = batch["value_target"].to(device)

        optimizer.zero_grad()

        # Forward pass with AMP
        if use_amp:
            with autocast(dtype=dtype):
                policy_logits, value_pred = model(board_tensor)
                loss, policy_loss, value_loss = loss_fn(
                    policy_logits, value_pred, policy_target, value_target
                )

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = model(board_tensor)
            loss, policy_loss, value_loss = loss_fn(
                policy_logits, value_pred, policy_target, value_target
            )
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train chess neural network")
    parser.add_argument("--config", type=str, required=True, help="Config file (main.yaml/mini.yaml)")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL")
    parser.add_argument("--val-data", type=str, help="Validation data JSONL")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--value-weight", type=float, default=0.25, help="Value loss weight")
    parser.add_argument("--amp", type=str, choices=["bf16", "fp16", "none"], default="bf16", help="AMP dtype")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", type=str, default="ckpts", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--max-samples", type=int, help="Max training samples")
    args = parser.parse_args()

    # Create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {config}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")

    # Create model
    if config["model_size"] == "main":
        model = create_model_main(input_channels=config["input_channels"])
    else:
        model = create_model_mini(input_channels=config["input_channels"])

    model = model.to(args.device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    # Create datasets
    train_dataset = ChessPositionDataset(
        args.data,
        augment=True,
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_dataset = ChessPositionDataset(args.val_data, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # Loss function
    loss_fn = ChessLoss(value_weight=args.value_weight)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        start_time = time.time()
        metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=args.device,
            use_amp=(args.amp != "none"),
            amp_dtype=args.amp,
        )
        epoch_time = time.time() - start_time

        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Validation
        if val_loader:
            model.eval()
            val_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    board_tensor = batch["board_tensor"].to(args.device)
                    policy_target = batch["policy_target"].to(args.device)
                    value_target = batch["value_target"].to(args.device)

                    policy_logits, value_pred = model(board_tensor)
                    loss, policy_loss, value_loss = loss_fn(
                        policy_logits, value_pred, policy_target, value_target
                    )

                    val_metrics["loss"] += loss.item()
                    val_metrics["policy_loss"] += policy_loss.item()
                    val_metrics["value_loss"] += value_loss.item()
                    num_batches += 1

            val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

        # Save checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": metrics["loss"],
            "config": config,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Update scheduler
        scheduler.step()

    # Save final model
    final_path = ckpt_dir / "latest.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
