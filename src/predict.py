import sys, torch
from PIL import Image
from torchvision import transforms
from config import OUT_DIR, IMG_SIZE
from model import build_model

img_path = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "dual"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(name).to(device)
model.load_state_dict(torch.load(OUT_DIR / f"best_{name}.pt", map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])

img = Image.open(img_path).convert("RGB")
with torch.no_grad():
    p_real = torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).item()

label = "REAL" if p_real > 0.5 else "AI-GENERATED"
print(f"{label} (P(real) = {p_real:.3f})")