# 🧠 Real-Time Human Emotion Recognition using Deep CNNs

**Project for Human-Computer Interaction / Deep Learning Course**  
**Group Members:** Abdur Rahman Goraya (2022035) · Ahsan Naqvi (2022073)

This project trains, evaluates, and deploys facial emotion-recognition models on the FER-2013 dataset. It includes:

- a lightweight custom baseline CNN for fast grayscale inference,
- a MobileNetV2 transfer-learning model for a stronger accuracy-oriented baseline,
- offline evaluation plots and classification reports,
- offline testing and benchmarking, and
- a Flask web interface for image upload and API-style prediction.

---

## 📌 Quick Start

```bash
cd /home/arg/Documents/her-dl
pip install -r requirements.txt
python train_model.py --data fer2013 --epochs 100 --batch_size 64
python test_model.py --model models/best_model.keras --data fer2013
python app.py
```

If you use an AMD ROCm setup that needs a graphics override, export it before training or inference:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

For full installation, hyperparameter, training, evaluation, and troubleshooting notes, see [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md).

---

## 📊 Baseline Model Performance

The saved baseline CNN checkpoint currently reports the following metadata:

| Item | Value |
|---|---:|
| Model | `EmotionNet_MobileInspired` |
| Input | `48×48×1` grayscale |
| Best validation accuracy | **59.70%** |
| Test loss | **1.1466** |
| Epochs run | **64** |
| Batch size | **128** |
| Learning rate | **0.001** |
| Augmentation | Enabled |
| Class weights | Disabled |
| Checkpoint | `models/best_model.keras` |

Generated baseline summary image:

![Baseline CNN performance summary](logs/baseline_performance_summary.png)

Generated evaluation plots from `test_model.py`:

![Baseline confusion matrix](logs/primary_model_confusion_matrix.png)

![Baseline per-class metrics](logs/primary_model_per_class_metrics.png)

The matching text report is saved at `logs/primary_model_classification_report.txt`.

> Note: `fer2013/` is currently empty in this checkout, so any future re-run of the evaluation pipeline will require the FER-2013 dataset in the expected folder structure. Running `python test_model.py --model models/best_model.keras --data fer2013` will regenerate the same baseline evaluation artifacts in `logs/`.

---

## 📋 Background Research Summary

### 1. FER-2013 Dataset

FER-2013 was introduced as part of the ICML 2013 Challenges in Representation Learning competition. It is a difficult “in-the-wild” facial-expression dataset because images are low-resolution, noisy, imbalanced, and collected from uncontrolled web sources.

**Dataset characteristics:**

- **35,887 total images**: 28,709 training, 3,589 public test, 3,589 private test.
- **48×48 grayscale images**.
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Strong class imbalance**: Happy is much more common than Disgust.
- **Label noise**: labels were collected through crowd annotation, so many examples are ambiguous or mislabeled.
- **Human-level reference**: commonly reported human accuracy is about **65.5%**.

This makes FER-2013 a useful benchmark for HCI emotion-recognition experiments because it reflects many real-world difficulties: lighting variation, pose variation, occlusion, low resolution, ambiguous expressions, and class imbalance.

### 2. CNNs for Facial Expression Recognition

Older facial-expression systems often used hand-crafted features such as Haar-like features, HOG descriptors, and SVM classifiers. CNNs improved performance by learning features directly from image pixels:

- early layers learn edges and simple texture patterns,
- middle layers learn facial parts such as eyes, brows, and mouth corners,
- deeper layers combine these features into expression-level representations.

Representative FER-2013 results from the literature include:

| Model / Method | Approx. Accuracy | Notes |
|---|---:|---|
| Traditional SVM baseline | ~44% | Hand-crafted features |
| Tang SVM-loss CNN | 71.2% | ICML 2013 winning approach |
| VGG-style CNNs | ~72% | Larger deep CNNs |
| Ensemble CNNs | ~74% | Multiple models combined |
| EfficientNet-style models | 76%+ | Modern transfer-learning families |
| Human reference | ~65.5% | Dataset is noisy and subjective |

### 3. MobileNet and Depthwise Separable Convolutions

The custom baseline CNN is inspired by MobileNet’s efficient depthwise separable convolutions. Instead of applying a full convolution that mixes spatial and channel information at the same time, a depthwise separable block splits the operation into:

1. **Depthwise convolution**: one spatial filter per input channel.
2. **Pointwise convolution**: a `1×1` convolution that mixes channels.

This reduces computation substantially compared with standard convolution and makes the baseline suitable for low-latency local or web-app inference.

### 4. Affective Computing and HCI Motivation

Facial emotion recognition is part of affective computing: building systems that can detect, interpret, or adapt to human emotional signals. Potential HCI applications include:

- adaptive tutoring systems,
- mental-health and mood monitoring,
- accessibility tools,
- customer-service analytics,
- human-robot interaction, and
- context-aware user interfaces.

---

## 🏗️ Project Structure

```text
her-dl/
├── app.py                         # Flask web application
├── data_loader.py                 # FER-2013 loading, preprocessing, augmentation
├── model.py                       # Baseline CNN and MobileNetV2 transfer model
├── train_model.py                 # Baseline CNN training script
├── train_transfer.py              # Two-stage transfer-learning script
├── test_model.py                  # Evaluation, comparison, and benchmark script
├── requirements.txt               # Python dependencies
├── SETUP_AND_RUN.md               # Detailed run guide and hyperparameter reference
├── fer2013/                       # Dataset directory: train/ and test/ class folders
├── models/                        # Saved .keras checkpoints and metadata sidecars
├── logs/                          # Training/evaluation plots and reports
├── screenshots/                   # Optional saved prediction screenshots
└── templates/
    └── index.html                 # Flask UI template
```

---

## 🧪 Baseline CNN Architecture

The baseline model is `EmotionNet_MobileInspired`, a compact grayscale CNN using standard convolution blocks followed by depthwise separable convolution blocks.

```text
Input: 48×48×1 grayscale face normalized to [0, 1]
│
├── Conv2D(32, 3×3) → BatchNorm → ReLU6
├── Conv2D(32, 3×3) → BatchNorm → ReLU6
├── MaxPool2D
│
├── DepthwiseSeparableConv(64)  → Dropout(0.10)
├── DepthwiseSeparableConv(64)  → Dropout(0.10)
├── MaxPool2D
│
├── DepthwiseSeparableConv(128) → Dropout(0.15)
├── DepthwiseSeparableConv(128) → Dropout(0.15)
├── MaxPool2D
│
├── DepthwiseSeparableConv(256) → Dropout(0.20)
├── DepthwiseSeparableConv(256) → Dropout(0.20)
├── MaxPool2D
│
├── DepthwiseSeparableConv(512) → Dropout(0.25)
├── GlobalAveragePooling2D
│
├── Dense(256) → BatchNorm → Dropout(0.50)
├── Dense(128) → BatchNorm → Dropout(0.30)
└── Dense(7, Softmax)
```

**Regularization and optimization choices:**

- L2 weight decay of `1e-4` in convolution and dense layers.
- Progressive dropout from `0.10` in early feature blocks to `0.50` in the classifier head.
- Adam optimizer with configurable learning rate.
- Optional label smoothing for noisy labels.
- Optional class weighting for FER-2013 class imbalance.
- Early stopping and learning-rate reduction on plateau.

---

## 🔁 Transfer-Learning Architecture

The transfer-learning path builds `EmotionNet_MobileNetV2`:

```text
Input: 96×96×3 RGB image
│
├── MobileNetV2 backbone, ImageNet weights, include_top=False
├── GlobalAveragePooling2D
├── BatchNormalization
├── Dropout(0.20)
├── Dense(256, ReLU, L2=1e-4)
├── BatchNormalization
├── Dropout(0.30)
└── Dense(7, Softmax)
```

Training is done in two stages:

1. **Head training**: freeze MobileNetV2 and train only the new classifier head.
2. **Fine tuning**: unfreeze the top MobileNetV2 layers and continue training with a much smaller learning rate.

---

## 📂 Dataset Layout

Download FER-2013 from Kaggle and place it in `fer2013/`:

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Kaggle dataset link: <https://www.kaggle.com/datasets/msambare/fer2013>

---

## 🚀 Common Commands

### Train the baseline CNN

```bash
python train_model.py --data fer2013 --epochs 100 --batch_size 64
```

### Train the transfer-learning model

```bash
python train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --model_path models/transfer_best_model.keras
```

### Evaluate one model

```bash
python test_model.py --model models/best_model.keras --data fer2013
```

### Compare baseline and transfer-learning models

```bash
python test_model.py --model models/best_model.keras --compare_model models/transfer_best_model.keras --data fer2013
```

### Launch the Flask app

```bash
python app.py
```

Then open <http://localhost:5000>.

---

## 🔬 Inference Pipeline

```text
Camera or uploaded image
        │
        ▼
OpenCV image loading / webcam frame capture
        │
        ▼
Face preprocessing
  - grayscale path for baseline CNN
  - RGB + MobileNetV2 preprocessing for transfer model
        │
        ▼
Resize to model input size
  - baseline: 48×48×1
  - transfer: 96×96×3
        │
        ▼
Model forward pass
        │
        ▼
Softmax probabilities over 7 emotions
        │
        ▼
Predicted emotion, confidence score, and optional visualization
```

The runtime loads metadata sidecars such as `models/best_model.metadata.json` and `models/transfer_best_model.metadata.json`, allowing the same inference scripts to adapt to different model input sizes and preprocessing schemes.

---

## 📈 Expected Results

FER-2013 is noisy and challenging, so performance varies with preprocessing, augmentation, class weighting, learning rate, and training duration.

| Model | Input | Expected Strength | Tradeoff |
|---|---|---|---|
| Baseline CNN | `48×48×1` | Fast and lightweight | Lower ceiling than large transfer models |
| MobileNetV2 transfer model | `96×96×3` | Better feature reuse from ImageNet | Slower and heavier |

A well-trained CNN on FER-2013 often aims for roughly human-level performance near **65%** top-1 accuracy, with higher top-2 accuracy because some labels are ambiguous even to humans.

---

## 📚 References

1. Goodfellow, I. J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests.* arXiv:1307.0414.
2. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.
3. Simonyan, K. and Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.
4. Viola, P. and Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR.
5. Ekman, P. and Friesen, W. V. (1971). *Constants across cultures in the face and emotion.* Journal of Personality and Social Psychology.
6. Picard, R. W. (1997). *Affective Computing.* MIT Press.

---

## 🤝 Acknowledgments

- Dataset: FER-2013 / Kaggle / Goodfellow et al.
- Frameworks: TensorFlow, Keras, scikit-learn, OpenCV, Flask.
- Architecture inspiration: MobileNet and efficient CNN design.
