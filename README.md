# AI-Generated Image Detection using CNN

Classifies images as **real** or **AI-generated** with a CNN, using frequency (FFT), noise and texture analysis.

## Models
- **Baseline CNN**: spatial (RGB) branch only
- **Dual-branch CNN**: RGB branch + FFT-spectrum branch

## Dataset
[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (32x32 images).
Download it and arrange it like this:

```
data/
├── train/ (FAKE/, REAL/)
└── test/  (FAKE/, REAL/)
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python src/analysis.py            # FFT / noise / texture analysis
python src/train.py simple        # train baseline
python src/train.py dual          # train dual-branch
python src/evaluate.py simple     # metrics, confusion matrix, ROC
python src/evaluate.py dual
python src/compare.py             # comparison table
python src/predict.py path/to/image.jpg dual
streamlit run src/app.py          # web demo
```

## Results
<!-- RESULTS_START -->
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Parameters |
|---|---|---|---|---|---|---|
| Baseline CNN (spatial) | 95.24% | 94.59% | 95.97% | 95.27% | 0.9908 | 102,017 |
| Dual-branch CNN (spatial + FFT) | 95.08% | 95.38% | 94.74% | 95.06% | 0.9893 | 203,329 |
<!-- RESULTS_END -->


    Directory: D:\ai-image-detector\outputs


Mode                 LastWriteTime         Length Name                                                                               
----                 -------------         ------ ----                                                                               
-a----        10-11-2026     05:19            192 comparison.csv                                                                     

## Limitations
Trained on CIFAKE only (32x32 images, one generator). Performance drops on high-resolution images and other generators such as Midjourney or DALL-E.

## Tech stack
Python, PyTorch, OpenCV, scikit-image, scikit-learn, Streamlit