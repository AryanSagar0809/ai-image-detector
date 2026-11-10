def evaluate(name):
    model = build_model(name).to(device)
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_{name}.pt", map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = torch.sigmoid(model(x.to(device)).squeeze(1).float()).cpu().numpy()
            probs.extend(p); labels.extend(y.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs > 0.5).astype(int)

    print(f"===== {name} model =====")
    print(classification_report(labels, preds, target_names=classes, digits=4))
    print("ROC-AUC:", roc_auc_score(labels, probs))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix ({name})")
    plt.savefig(f"{OUT_DIR}/cm_{name}.png", dpi=150); plt.show()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"ROC ({name})")
    plt.savefig(f"{OUT_DIR}/roc_{name}.png", dpi=150); plt.show()

evaluate("simple")
evaluate("dual")