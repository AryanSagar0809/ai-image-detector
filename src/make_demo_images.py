import random, shutil, torch
from PIL import Image
from torchvision import transforms
from config import DATA_DIR, OUT_DIR, BASE_DIR, IMG_SIZE
from model import build_model

MODEL_NAME = "dual"
POOL = 300            # images sampled per class
KEEP_CORRECT = 5      # confident correct examples per class
KEEP_WRONG = 4        # mistakes, for an honest discussion

model = build_model(MODEL_NAME)
model.load_state_dict(torch.load(OUT_DIR / f"best_{MODEL_NAME}.pt", map_location="cpu"))
model.eval()

tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

dest = BASE_DIR / "demo_images"
for sub in ["REAL_correct", "FAKE_correct", "model_mistakes"]:
    (dest / sub).mkdir(parents=True, exist_ok=True)

random.seed(7)
found = {"REAL_correct": [], "FAKE_correct": [], "model_mistakes": []}

for cls in ["REAL", "FAKE"]:
    files = random.sample(list((DATA_DIR / "test" / cls).glob("*")), POOL)
    for f in files:
        img = Image.open(f).convert("RGB")
        with torch.no_grad():
            p_real = torch.sigmoid(model(tf(img).unsqueeze(0))).item()
        pred = "REAL" if p_real > 0.5 else "FAKE"
        conf = p_real if pred == "REAL" else 1 - p_real
        key = f"{cls}_correct" if pred == cls else "model_mistakes"
        found[key].append((conf, f, cls, pred))

for key, items in found.items():
    items.sort(key=lambda t: -t[0])
    keep = KEEP_WRONG if key == "model_mistakes" else KEEP_CORRECT
    for conf, f, cls, pred in items[:keep]:
        name = (f"true-{cls}_pred-{pred}_{conf:.2f}_{f.name}" if key == "model_mistakes"
                else f"{cls}_{conf:.2f}_{f.name}")
        shutil.copy(f, dest / key / name)
    print(f"{key}: copied {min(keep, len(items))} images")

print("Done ->", dest)