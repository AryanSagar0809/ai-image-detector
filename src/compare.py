import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import OUT_DIR
from dataset import get_loaders
from model import build_model


def get_scores(name, test_loader, device):
    model = build_model(name).to(device)
    model.load_state_dict(torch.load(OUT_DIR / f"best_{name}.pt", map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy()
            probs.extend(p); labels.extend(y.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs > 0.5).astype(int)
    return {"Model": name,
            "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds),
            "Recall": recall_score(labels, preds),
            "F1": f1_score(labels, preds),
            "ROC-AUC": roc_auc_score(labels, probs),
            "Params": sum(p.numel() for p in model.parameters())}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, _ = get_loaders()
    df = pd.DataFrame([get_scores(n, test_loader, device) for n in ["simple", "dual"]]).round(4)
    df["Model"] = df["Model"].map({"simple": "Baseline CNN (spatial)",
                                   "dual": "Dual-branch CNN (spatial + FFT)"})
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "comparison.csv", index=False)


if __name__ == "__main__":
    main()