import base64
import io

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from config import OUT_DIR, IMG_SIZE
from model import build_model

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Style inspired by schematik.io: warm off-white canvas with a faint dot grid,
# huge bold headline, black pill buttons, rounded white cards, one orange accent.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #faf8f4;
    --ink: #111111;
    --muted: #6b675e;
    --line: #ebe6da;
    --card: #ffffff;
    --accent: #ff6a3d;
    --good: #16a34a;
    --bad: #e5484d;
}

html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
    background-color: var(--bg);
    background-image: radial-gradient(#ddd6c6 1px, transparent 1px);
    background-size: 24px 24px;
    color: var(--ink);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 980px; padding-top: 1rem; padding-bottom: 3rem; }

.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* ---------- Announcement bar ---------- */
.banner {
    text-align: center; font-size: 0.85rem; font-weight: 600;
    background: var(--ink); color: #fff; padding: 8px 14px; border-radius: 999px;
    margin: 0 auto 1.2rem auto; width: fit-content;
}
.banner b { color: var(--accent); }

/* ---------- Nav ---------- */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,0.85); border: 1px solid var(--line);
    border-radius: 999px; padding: 10px 14px 10px 24px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05); margin-bottom: 3rem;
}
.nav .brand { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; }
.nav .brand span { color: var(--accent); }
.nav .links { display: flex; gap: 1.6rem; font-weight: 600; font-size: 0.95rem; color: var(--muted); }
.nav .try {
    background: var(--ink); color: #fff; font-weight: 700; font-size: 0.9rem;
    padding: 9px 20px; border-radius: 999px;
}

/* ---------- Hero ---------- */
.hero { text-align: center; margin-bottom: 2.2rem; }
.hero h1 {
    font-size: clamp(2.6rem, 7vw, 4.6rem); font-weight: 800;
    letter-spacing: -0.045em; line-height: 1.02; margin: 0 0 1rem 0; color: var(--ink);
}
.hero h1 em {
    font-style: normal; background: linear-gradient(transparent 62%, rgba(255,106,61,0.35) 62%);
    padding: 0 6px;
}
.hero p.sub { font-size: 1.2rem; color: var(--muted) !important; margin: 0 0 1.2rem 0; }
.tags { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
.tag {
    padding: 6px 16px; border-radius: 999px; background: #fff; border: 1px solid var(--line);
    font-weight: 600; font-size: 0.9rem; color: var(--ink);
}

/* ---------- Cards / headings ---------- */
.card {
    background: var(--card); border: 1px solid var(--line); border-radius: 24px;
    padding: 1.3rem 1.5rem; box-shadow: 0 6px 24px rgba(0,0,0,0.05); margin-bottom: 1rem;
}
.card h4 { margin: 0 0 0.7rem 0; font-size: 1.05rem; font-weight: 700; }
.section-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin: 3rem 0 0.3rem 0; color: var(--ink);
}
.section-sub { color: var(--muted) !important; margin-bottom: 1.2rem; }

/* ---------- Prompt-style upload box ---------- */
.prompt-label { font-weight: 700; font-size: 1.15rem; margin: 0 0 0.5rem 0.4rem; }
[data-testid="stFileUploader"] section {
    background: #fff; border: 2px dashed #d9d2c1; border-radius: 24px; padding: 1.4rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.06);
}
[data-testid="stFileUploader"] section * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button {
    background: var(--ink); color: #fff !important; border: none; border-radius: 999px;
    padding: 0.45rem 1.4rem; font-weight: 700;
}
[data-testid="stFileUploader"] button * { color: #fff !important; }

/* ---------- Pill radio ---------- */
div[role="radiogroup"] { gap: 10px; }
div[role="radiogroup"] label {
    background: #fff; border: 1px solid var(--line); border-radius: 999px;
    padding: 6px 16px; cursor: pointer;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label:has(input:checked) { background: var(--ink); border-color: var(--ink); }
div[role="radiogroup"] label:has(input:checked) * { color: #fff !important; }

/* ---------- Verdict ---------- */
.verdict { border-radius: 24px; padding: 1.5rem 1.6rem; border: 1px solid var(--line); background: #fff;
           box-shadow: 0 8px 28px rgba(0,0,0,0.07); }
.verdict .pill {
    display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 800;
    font-size: 0.8rem; letter-spacing: 0.08em; color: #fff;
}
.verdict.real .pill { background: var(--good); }
.verdict.fake .pill { background: var(--bad); }
.verdict h2 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.035em; margin: 0.7rem 0 0.2rem 0; line-height: 1.05; }
.verdict p { color: var(--muted) !important; margin: 0; }

/* ---------- Bars ---------- */
.bar-row { display: flex; justify-content: space-between; font-weight: 600; font-size: 0.92rem; }
.bar { height: 12px; border-radius: 999px; background: #f0ebe0; overflow: hidden; margin: 6px 0 14px 0; }
.bar > div { height: 100%; border-radius: 999px; }
.bar .real { background: var(--good); }
.bar .fake { background: var(--bad); }

/* ---------- Chips ---------- */
.chip {
    display: inline-block; padding: 4px 12px; margin: 4px 6px 0 0; border-radius: 999px;
    background: #f5f1e8; color: var(--ink); font-size: 0.8rem; font-weight: 600; border: 1px solid var(--line);
}

/* ---------- Metrics, buttons, expander ---------- */
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 0.7rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -0.02em; }
.stButton button {
    background: var(--ink); color: #fff; border: none; border-radius: 999px; font-weight: 700; padding: 0.5rem 1.4rem;
}
.stButton button:hover { background: var(--accent); color: #fff; }
[data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); border-radius: 20px; }
img { border-radius: 16px; }

/* ---------- Scan gallery (like template cards) ---------- */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }
.scan { background: #fff; border: 1px solid var(--line); border-radius: 20px; overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05); }
.scan img { width: 100%; height: 150px; object-fit: cover; border-radius: 0; display: block; }
.scan .meta { padding: 10px 14px 14px 14px; }
.scan .cat { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; }
.scan .cat.real { color: var(--good); }
.scan .cat.fake { color: var(--bad); }
.scan .name { font-weight: 700; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scan .pr { color: var(--muted); font-size: 0.82rem; }

/* ---------- Steps ---------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
.step { background: #fff; border: 1px solid var(--line); border-radius: 24px; padding: 1.4rem 1.5rem; }
.step .n { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em; color: var(--accent); }
.step h3 { margin: 0.3rem 0 0.4rem 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
.step p { margin: 0; color: var(--muted) !important; }

/* ---------- Footer ---------- */
.foot { text-align: center; margin-top: 4.5rem; padding-top: 2rem; border-top: 1px solid var(--line); }
.foot h2 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 0.6rem 0; }
.foot p { color: var(--muted) !important; font-size: 0.9rem; margin: 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource
def load(name):
    m = build_model(name)
    m.load_state_dict(torch.load(OUT_DIR / f"best_{name}.pt", map_location="cpu"))
    return m.eval()


tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def predict(img, name):
    with torch.no_grad():
        return torch.sigmoid(load(name)(tf(img).unsqueeze(0))).item()


def fft_spectrum(img):
    """Log-magnitude FFT spectrum, tinted with the page's orange accent."""
    gray = np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    mag = (255 * (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)).astype(np.uint8)
    return ImageOps.colorize(Image.fromarray(mag), black="#111111", mid="#ff6a3d", white="#fff4e6")


def thumb_b64(img, size=360):
    t = img.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# Top: banner + nav + hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="banner">New: dual-branch model that checks <b>frequency artefacts</b> too</div>
<div class="nav">
<div class="brand">/ detector<span>.</span></div>
<div class="links"><span>Detect</span><span>How it works</span><span>About</span></div>
<div class="try">Try now</div>
</div>
<div class="hero">
<h1>Spot <em>AI images</em><br>in seconds.</h1>
<p class="sub">Drop in a picture. Get a verdict, a confidence score and a look inside the pixels.</p>
<div class="tags"><span class="tag">CNN</span><span class="tag">FFT branch</span><span class="tag">CIFAKE</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Prompt-style input
# ----------------------------------------------------------------------------
st.markdown('<div class="prompt-label">What are we checking today?</div>', unsafe_allow_html=True)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with st.expander("⚙️ Settings"):
    model_name = st.radio(
        "Model",
        ["dual", "simple"],
        horizontal=True,
        format_func=lambda n: "Dual-branch (spatial + FFT)" if n == "dual" else "Baseline CNN",
    )
    c_a, c_b = st.columns(2)
    threshold = c_a.slider("Decision threshold, P(real)", 0.10, 0.90, 0.50, 0.05,
                           help="Images with P(real) above this are labelled REAL.")
    show_fft = c_b.toggle("Show FFT spectrum", value=True)

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
if file:
    img = Image.open(file).convert("RGB")
    with st.spinner("Analysing image..."):
        p_real = predict(img, model_name)

    is_real = p_real > threshold
    conf = p_real if is_real else 1 - p_real
    p_fake = 1 - p_real

    key = f"{file.name}-{file.size}-{model_name}"
    if st.session_state.get("last_key") != key:
        st.session_state.last_key = key
        st.session_state.history.append(
            {"name": file.name, "p": p_real, "real": is_real, "thumb": thumb_b64(img)}
        )

    st.markdown('<div class="section-title">Your result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Here is what the model thinks.</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h4>Preview</h4>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(
            f'<span class="chip">{file.name}</span>'
            f'<span class="chip">{img.width}×{img.height}px</span>'
            f'<span class="chip">{file.size / 1024:.1f} KB</span>'
            f'<span class="chip">Model input {IMG_SIZE}×{IMG_SIZE}</span></div>',
            unsafe_allow_html=True,
        )

    with right:
        if is_real:
            cls, tag, head = "real", "REAL", "Looks like a real photo."
        else:
            cls, tag, head = "fake", "AI-GENERATED", "Looks AI-generated."
        st.markdown(
            f'<div class="verdict {cls}"><span class="pill">{tag}</span>'
            f"<h2>{head}</h2><p>Confidence: <b>{conf:.1%}</b></p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="card" style="margin-top:1rem;">
<h4>Probability breakdown</h4>
<div class="bar-row"><span>Real</span><span>{p_real:.1%}</span></div>
<div class="bar"><div class="real" style="width:{p_real * 100:.1f}%"></div></div>
<div class="bar-row"><span>AI-generated</span><span>{p_fake:.1%}</span></div>
<div class="bar"><div class="fake" style="width:{p_fake * 100:.1f}%"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("P(real)", f"{p_real:.3f}")
        m2.metric("P(AI)", f"{p_fake:.3f}")
        m3.metric("Threshold", f"{threshold:.2f}")

        if conf < 0.65:
            st.warning("Low confidence. The model is unsure about this image, so treat the result with caution.")

    if show_fft:
        st.markdown('<div class="section-title">Inside the pixels</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">AI images often leave periodic artefacts in the frequency '
            "spectrum. This is what the FFT branch looks at.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="large")
        c1.image(img.resize((IMG_SIZE * 3, IMG_SIZE * 3)), caption="Spatial view", use_container_width=True)
        c2.image(fft_spectrum(img).resize((IMG_SIZE * 3, IMG_SIZE * 3)),
                 caption="FFT log-magnitude spectrum", use_container_width=True)

# ----------------------------------------------------------------------------
# Recent scans gallery
# ----------------------------------------------------------------------------
if st.session_state.history:
    st.markdown('<div class="section-title">Recent scans</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Everything you checked in this session.</div>', unsafe_allow_html=True)
    cards = ""
    for h in reversed(st.session_state.history[-8:]):
        cls, cat = ("real", "REAL") if h["real"] else ("fake", "AI-GENERATED")
        cards += (
            f'<div class="scan"><img src="data:image/jpeg;base64,{h["thumb"]}"/>'
            f'<div class="meta"><div class="cat {cls}">{cat}</div>'
            f'<div class="name">{h["name"]}</div>'
            f'<div class="pr">P(real) = {h["p"]:.2f}</div></div></div>'
        )
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# How it works
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">From upload to verdict</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">No ML degree required.</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="steps">
<div class="step"><div class="n">STEP 1</div><h3>Upload</h3><p>Drop in any JPG or PNG. It is resized and normalised for the model.</p></div>
<div class="step"><div class="n">STEP 2</div><h3>Analyse</h3><p>A CNN reads textures and edges, while the FFT branch looks for frequency artefacts.</p></div>
<div class="step"><div class="n">STEP 3</div><h3>Verdict</h3><p>You get P(real), a label and a confidence score. Adjust the threshold in Settings.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="foot">
<h2>Making fakes<br>less sneaky.</h2>
<p>Trained on CIFAKE (32×32 images), so results on very different or high-resolution images may be less reliable.
This is a probabilistic tool, not proof.</p>
<p style="margin-top:0.6rem;">Built with PyTorch and Streamlit.</p>
</div>
""",
    unsafe_allow_html=True,
)