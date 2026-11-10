import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS


def get_loaders(subset=0, val_split=0.1, seed=42):
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), norm])

    train_full = datasets.ImageFolder(str(DATA_DIR / "train"), train_tf)
    val_full = datasets.ImageFolder(str(DATA_DIR / "train"), eval_tf)
    test_set = datasets.ImageFolder(str(DATA_DIR / "test"), eval_tf)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_full), generator=g).tolist()
    n_val = int(len(idx) * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if subset and subset < len(train_idx):
        train_idx = train_idx[:subset]
        val_idx = val_idx[:max(2000, subset // 10)]

    kw = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(Subset(train_full, train_idx), BATCH_SIZE, shuffle=True, **kw)
    val_loader = DataLoader(Subset(val_full, val_idx), BATCH_SIZE, **kw)
    test_loader = DataLoader(test_set, BATCH_SIZE, **kw)
    return train_loader, val_loader, test_loader, train_full.classes