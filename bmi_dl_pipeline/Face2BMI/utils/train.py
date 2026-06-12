
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def mixup_data(images, labels, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = images.size(0)
    # Keep permutation on CPU to avoid CUDA kernel issues
    index = torch.randperm(batch_size)
    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_images, mixed_labels


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad.clone() * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm

    def step(self, closure=None):
        pass


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_mae):
        score = -val_mae
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter    = 0


def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, ema=None,
                    use_sam=True, use_mixup=True):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            images, labels = mixup_data(images, labels, alpha=0.2)

        if use_sam:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.first_step(zero_grad=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    mae      = np.mean(np.abs(np.array(labels_all) -
                               np.array(preds_all)))
    return avg_loss, mae


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds_all.extend(outputs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    mae      = np.mean(np.abs(np.array(labels_all) -
                               np.array(preds_all)))
    return avg_loss, mae, np.array(preds_all), np.array(labels_all)
