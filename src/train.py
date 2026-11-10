import os, time, argparse
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import OUT_DIR
from dataset import get_loaders
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["simple", "dual"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--subset", type=int, default=0, help="0 = full training set")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {args.model} | subset: {args.subset or 'full'}")

    train_loader, val_loader, _, classes = get_loaders(subset=args.subset)
    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, correct, n = 0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, leave=False):
                x, y = x.to(device), y.float().to(device)
                out = model(x).squeeze(1)
                loss = criterion(out, y)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * len(y)
                correct += ((out > 0).float() == y).sum().item()
                n += len(y)
        return total_loss / n, correct / n

    hist = {"tl": [], "vl": [], "ta": [], "va": []}
    best_val, patience, bad = 0, 3, 0
    ckpt = OUT_DIR / f"best_{args.model}.pt"

    for epoch in range(args.epochs):
        t0 = time.time()
        tl, ta = run_epoch(train_loader, True)
        vl, va = run_epoch(val_loader, False)
        scheduler.step(vl)
        for k, v in zip(["tl", "vl", "ta", "va"], [tl, vl, ta, va]):
            hist[k].append(v)
        print(f"Epoch {epoch+1}/{args.epochs}: train {ta:.4f} | val {va:.4f} | {time.time()-t0:.0f}s")
        if va > best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping"); break

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["tl"], label="train"); ax[0].plot(hist["vl"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["ta"], label="train"); ax[1].plot(hist["va"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.savefig(OUT_DIR / f"curves_{args.model}.png", dpi=150)
    print(f"Best val acc: {best_val:.4f} | saved {ckpt}")


if __name__ == "__main__":
    main()