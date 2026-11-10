import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, OUT_DIR

README = BASE_DIR / "README.md"
START, END = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"

df = pd.read_csv(OUT_DIR / "comparison.csv")

lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Parameters |",
         "|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append(f"| {r['Model']} | {r['Accuracy']:.2%} | {r['Precision']:.2%} | "
                 f"{r['Recall']:.2%} | {r['F1']:.2%} | {r['ROC-AUC']:.4f} | {int(r['Params']):,} |")
table = "\n".join(lines)

text = README.read_text(encoding="utf-8")
if START not in text or END not in text:
    raise SystemExit("Markers not found in README.md. Add them as described in step 2a.")

before = text.split(START)[0]
after = text.split(END)[1]
README.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
print("README results table updated.\n")
print(table)