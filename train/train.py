import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from model import create_model, resolve_device
from train.dataset import ChessPositionDataset


class ChessLoss(nn.Module):
    def __init__(self, policy_weight=1., q_value_weight=0., value_weight=.25):
        super().__init__()
        self.policy_weight, self.q_value_weight, self.value_weight = policy_weight, q_value_weight, value_weight

    def forward(self, policy_logits, value_pred, policy_target, value_target,
                q_value_target=None, legal_mask=None, weights=None, value_mask=None, q_logits=None):
        batch = policy_logits.shape[0]
        logits, targets = policy_logits.float().reshape(batch, -1), policy_target.reshape(batch, -1)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask.reshape(batch, -1) == 0, torch.finfo(logits.dtype).min)
        weights = torch.ones(batch, device=logits.device) if weights is None else weights
        policy_rows = -(targets * F.log_softmax(logits, dim=1)).sum(1)
        policy_loss = (policy_rows * weights).sum() / weights.sum()
        q_loss = logits.new_zeros(())
        if q_value_target is not None and self.q_value_weight:
            valid = q_value_target >= 0
            # A separate head keeps absolute outcome calibration independent of policy log-odds.
            prediction = torch.sigmoid((policy_logits if q_logits is None else q_logits).float())
            q_targets = torch.where(valid, q_value_target, prediction.detach())
            counts = valid.reshape(batch, -1).sum(1)
            q_rows = ((prediction - q_targets).square()).reshape(batch, -1).sum(1) / counts.clamp_min(1)
            q_weights = weights * (counts > 0)
            q_loss = (q_rows * q_weights).sum() / q_weights.sum().clamp_min(1e-8)
        value_rows = (value_pred.float().reshape(-1) - value_target).square()
        value_weights = weights if value_mask is None else weights * value_mask
        value_loss = (value_rows * value_weights).sum() / value_weights.sum().clamp_min(1e-8)
        total = self.policy_weight * policy_loss + self.q_value_weight * q_loss + self.value_weight * value_loss
        return total, policy_loss, q_loss, value_loss


def train_epoch(model, dataloader, optimizer, loss_fn, device, use_amp=True,
                amp_dtype='bf16', scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    amp = use_amp and device.type in ('cuda', 'mps')
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, init_scale=1024 if device.type == 'mps' else 65536,
                                      enabled=amp and dtype == torch.float16)
    totals = np.zeros(4)
    correct, count = 0, 0
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=dtype, enabled=amp):
            policy, value, q_logits = model.forward_with_q(batch['board_tensor'])
            losses = loss_fn(policy, value, batch['policy_target'], batch['value_target'],
                             batch.get('q_value_target'), batch.get('legal_mask'), batch.get('weight'),
                             batch.get('value_mask'), q_logits)
        if not torch.isfinite(losses[0]):
            raise FloatingPointError('Non-finite training loss')
        if training:
            scaler.scale(losses[0]).backward()
            scaler.unscale_(optimizer)
            old_scale = scaler.get_scale()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            if scheduler and scaler.get_scale() >= old_scale:
                scheduler.step()
        n = policy.shape[0]
        totals += np.array([loss.detach().item() for loss in losses]) * n
        scores = policy.detach().flatten(1)
        if 'legal_mask' in batch:
            scores = scores.masked_fill(batch['legal_mask'].flatten(1) == 0, -torch.inf)
        correct += (scores.argmax(1) == batch['policy_target'].flatten(1).argmax(1)).sum().item()
        count += n
    if not count:
        raise ValueError('Empty training or validation dataset')
    return {**dict(zip(('loss', 'policy_loss', 'q_loss', 'value_loss'), (totals / count).tolist())),
            'top1': correct / count, 'samples': count}


def save_checkpoint(path, model, config, epoch, optimizer, scheduler, metrics):
    temporary = path.with_suffix('.tmp')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config,
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--data-weights')
    parser.add_argument('--val-data')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--policy-weight', type=float, default=1.)
    parser.add_argument('--q-value-weight', type=float, default=0.)
    parser.add_argument('--value-weight', type=float, default=.25)
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'none'], default='none')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--ckpt-dir', default='ckpts')
    parser.add_argument('--resume', help='Initial weights; --epochs is the number of additional epochs')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text())
    device = resolve_device(args.device)
    model = create_model(config).to(device)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        start = checkpoint.get('epoch', -1) + 1
    paths = args.data.split(',')
    weights = [float(w) for w in args.data_weights.split(',')] if args.data_weights else [1.] * len(paths)
    if len(weights) != len(paths) or any(not np.isfinite(w) or w <= 0 for w in weights):
        parser.error('--data-weights needs one positive finite weight per dataset')
    datasets = [ChessPositionDataset(path, max_samples=args.max_samples) for path in paths]
    for dataset, weight in zip(datasets, weights):
        dataset.weight = weight
    batch_size = args.batch_size or config.get('batch_size', 256)
    loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(ChessPositionDataset(args.val_data, augment=False), batch_size=batch_size) if args.val_data else None
    if not len(loader):
        parser.error('Empty training dataset')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or config.get('learning_rate', 1e-3), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(device.type, init_scale=1024 if device.type == 'mps' else 65536,
                                  enabled=args.amp == 'fp16' and device.type in ('cuda', 'mps'))
    loss_fn = ChessLoss(args.policy_weight, args.q_value_weight, args.value_weight)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f'{device}: {model.get_num_params():,} parameters, {len(loader.dataset):,} samples', flush=True)
    with (ckpt_dir / 'train.jsonl').open('a') as log:
        for epoch in range(start, start + args.epochs):
            begin = time.perf_counter()
            metrics = {'epoch': epoch + 1, 'train': train_epoch(model, loader, optimizer, loss_fn, device,
                       args.amp != 'none', args.amp, scheduler, scaler)}
            if val_loader:
                metrics['validation'] = train_epoch(model, val_loader, None, loss_fn, device, False)
            metrics['seconds'] = time.perf_counter() - begin
            print(json.dumps(metrics), flush=True)
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            save_checkpoint(ckpt_dir / f'epoch_{epoch + 1}.pt', model, config, epoch, optimizer, scheduler, metrics)
            save_checkpoint(ckpt_dir / 'latest.pt', model, config, epoch, optimizer, scheduler, metrics)


if __name__ == '__main__':
    main()
