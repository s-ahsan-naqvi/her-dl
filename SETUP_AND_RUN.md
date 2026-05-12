# ⚙️ Setup and Run Guide — Real-Time Human Emotion Recognition

## Important environment note

Some AMD / ROCm setups require an override before TensorFlow can use the GPU:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

If you do not need it, you can skip that line.

This guide covers:

- dependency installation,
- FER-2013 dataset layout,
- baseline CNN training,
- transfer-learning training,
- evaluation and model comparison,
- browser deployment, and
- troubleshooting.

---

## 1) Install dependencies

From the project root:

```bash
cd /home/arg/Documents/her-dl
pip install -r requirements.txt
```

If you use a virtual environment, activate it first.

### Optional cleanup / upgrade

If package installation fails, upgrade pip and retry:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) FER-2013 dataset layout

The scripts expect the Kaggle FER-2013 directory tree:

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

Download the dataset from Kaggle and place the extracted `fer2013/` folder in the project root.

Kaggle link:

```text
https://www.kaggle.com/datasets/msambare/fer2013
```

---

## 3) Baseline CNN training (`train_model.py`)

The baseline model is the lightweight grayscale CNN defined in `model.py`.
It trains on `48×48×1` images and is the fastest option for local inference.

### Basic training command

```bash
python train_model.py --data fer2013 --epochs 100 --batch_size 64
```

### What this command does

- loads FER-2013 from `train/` and `test/`,
- splits the training set into train/validation internally,
- normalizes grayscale inputs to `[0, 1]`,
- applies augmentation unless disabled,
- computes class weights unless disabled,
- trains with Adam,
- checkpoints the best model by the selected monitor metric,
- evaluates the final best checkpoint on the test set, and
- writes plots, history, and metadata files.

### Baseline hyperparameters and CLI options

| Argument | Type | Default | Meaning | Practical effect |
|---|---:|---:|---|---|
| `--data` | path | `fer2013` | Dataset root containing `train/` and `test/` | Required for real training |
| `--epochs` | int | `100` | Maximum number of epochs to run | Larger values allow longer training, but early stopping may stop sooner |
| `--batch_size` | int | `64` | Number of samples per gradient step | Larger batches are faster but use more memory |
| `--lr` | float | `1e-3` | Adam learning rate | Too high can destabilize training; too low can slow convergence |
| `--resume` | path | `None` | Resume from a `.keras` file or `.weights.h5` file | Useful for continuing an interrupted run |
| `--no_augment` | flag | off | Disable the image augmentation pipeline | Use only for debugging or ablation studies |
| `--include_brightness_augment` | flag | off | Add normalized brightness augmentation | Can help robustness to lighting variation |
| `--model_path` | path | `models/best_model.keras` | Final clean model save path | This is the file used later for evaluation and inference |
| `--no_savedmodel` | flag | off | Skip SavedModel export | Saves time if you only want `.keras` |
| `--savedmodel_path` | path | `models/emotion_model_savedmodel` | SavedModel export location | Only used if SavedModel export is enabled |
| `--no_tensorboard` | flag | off | Disable TensorBoard logging | Useful when TensorBoard is not installed |
| `--no_class_weights` | flag | off | Disable balanced class weights | Set this if you want unweighted training |
| `--label_smoothing` | float | `0.0` | Label smoothing for categorical cross-entropy | Helps reduce overconfidence on noisy labels |
| `--monitor` | str | `val_loss` | Metric used for checkpointing, early stopping, and LR reduction | Use `val_accuracy` if you want to maximize validation accuracy instead |
| `--overfit_samples` | int | `0` | Debug mode using the first `N` training samples | Good for checking whether the model can overfit a tiny subset |

### Baseline training behavior that is not exposed as CLI flags

These are still important hyperparameters because the script uses them internally:

| Component | Setting | Purpose |
|---|---|---|
| Optimizer | Adam | Stable default optimizer for CNN training |
| BatchNorm / ReLU | present in all blocks | Improves optimization stability and nonlinearity |
| Regularization | L2 weight decay = `1e-4` | Reduces overfitting |
| Dropout | `0.10 → 0.50` | Stronger regularization in deeper layers and classifier head |
| Early stopping patience | `15` epochs | Stops training when the monitored metric stops improving |
| Reduce-on-plateau patience | `5` epochs | Lowers the learning rate when progress stalls |
| Reduce-on-plateau factor | `0.5` | Halves the learning rate on plateau |
| Minimum LR | `1e-7` | Prevents the learning rate from shrinking too far |

### Recommended baseline settings

#### Fast smoke test

```bash
python train_model.py --data fer2013 --epochs 5 --batch_size 32 --no_class_weights --no_augment
```

#### Balanced default run

```bash
python train_model.py --data fer2013 --epochs 100 --batch_size 64
```

#### More regularized run for noisy data

```bash
python train_model.py --data fer2013 --epochs 120 --batch_size 64 --label_smoothing 0.1 --include_brightness_augment
```

#### Resume training

```bash
python train_model.py --data fer2013 --resume models/best_model.keras --epochs 150
```

### Baseline training outputs

After training, the script creates:

- `models/best_model.keras`
- `models/best_model.weights.h5`
- `models/best_model.metadata.json`
- `logs/training_curves.png`
- `logs/training_history.csv`
- `logs/training_metadata.json`
- `logs/emotion_model_savedmodel/` unless `--no_savedmodel` is used

---

## 4) Transfer-learning training (`train_transfer.py`)

The transfer-learning path uses `MobileNetV2` as a backbone and trains in two stages:

1. **Stage 1** — freeze the backbone and train the classifier head.
2. **Stage 2** — unfreeze the upper backbone layers and fine-tune with a smaller learning rate.

This path expects RGB input and uses MobileNetV2 preprocessing.

### Basic training command

```bash
python train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --model_path models/transfer_best_model.keras
```

### Transfer-learning hyperparameters and CLI options

| Argument | Type | Default | Meaning | Practical effect |
|---|---:|---:|---|---|
| `--data` | path | `fer2013` | Dataset root containing `train/` and `test/` | Required for real training |
| `--backbone` | str | `MobileNetV2` | Pretrained backbone architecture | Only `MobileNetV2` is currently implemented |
| `--input_size` | int | `96` | Input image size for the backbone | Larger than baseline to better match pretrained features |
| `--batch_size` | int | `32` | Batch size for both stages | Often smaller than baseline because RGB + MobileNet is heavier |
| `--head_epochs` | int | `10` | Stage 1 epochs with frozen backbone | Trains the classifier head first |
| `--fine_tune_epochs` | int | `10` | Stage 2 epochs for fine-tuning | Continues training after the head stabilizes |
| `--fine_tune_at` | int | `-30` | Layer index where fine-tuning begins | Negative values count from the end of the backbone |
| `--head_lr` | float | `1e-3` | Learning rate for Stage 1 | Good for the newly initialized head |
| `--fine_tune_lr` | float | `1e-5` | Learning rate for Stage 2 | Small LR prevents destroying pretrained features |
| `--weights` | str | `imagenet` | Backbone weight initialization | `imagenet` is the usual choice |
| `--model_path` | path | `models/transfer_best_model.keras` | Final model save path | Used later for evaluation and inference |
| `--log_dir` | path | `logs/transfer` | Directory for transfer-learning logs | Stores both stage histories and plots |
| `--no_class_weights` | flag | off | Disable balanced class weights | Useful if you want unweighted training |
| `--freeze_bn` | flag | on | Freeze BatchNorm layers during fine-tuning | Usually improves stability with small datasets |
| `--no_freeze_bn` | flag | off | Allow BatchNorm layers to update | Useful for experimentation, but riskier on small datasets |
| `--patience` | int | `8` | Early stopping patience for each stage | Stops a stage if validation stops improving |
| `--resume` | path | `None` | Resume from an existing `.keras` model | Useful for continuing a prior run |
| `--no_tensorboard` | flag | off | Disable TensorBoard logging | Useful if TensorBoard is unavailable |

### Transfer-learning stage behavior

| Stage | Learning rate | Backbone | Goal |
|---|---:|---|---|
| Stage 1 | `--head_lr` | Frozen | Learn the new classifier head |
| Stage 2 | `--fine_tune_lr` | Partially unfrozen | Adapt the pretrained features to FER-2013 |

### Fine-tuning guidance

- Use a **more negative** `--fine_tune_at` value to fine-tune fewer layers.
- Use a **less negative** value, or `0`, to fine-tune more layers.
- Keep `--freeze_bn` enabled unless you have a strong reason to let BatchNorm update.
- Use `--fine_tune_lr` much smaller than `--head_lr`.

### Recommended transfer-learning settings

#### Quick smoke test

```bash
python train_transfer.py --data fer2013 --head_epochs 3 --fine_tune_epochs 3 --batch_size 16 --weights imagenet
```

#### Balanced default run

```bash
python train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --batch_size 32
```

#### More conservative fine-tuning

```bash
python train_transfer.py --data fer2013 --head_epochs 8 --fine_tune_epochs 8 --fine_tune_at -50 --freeze_bn
```

#### Stronger regularization for noisy labels

```bash
python train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --no_class_weights
```

### Transfer-learning outputs

After training, the script creates:

- `models/transfer_best_model.keras`
- `models/transfer_best_model.metadata.json`
- `logs/transfer/training_curves.png`
- `logs/transfer/training_metadata.json`
- `logs/transfer/stage1/history.csv`
- `logs/transfer/stage2/history.csv`
- `logs/transfer/savedmodel/`

---

## 5) Testing and evaluation (`test_model.py`)

This script evaluates a saved `.keras` model, computes metrics, and generates plots.
It can test the baseline CNN, the transfer-learning model, or both side by side.

### Evaluate a single model

```bash
python test_model.py --model models/best_model.keras --data fer2013
```

### Evaluate the transfer model

```bash
python test_model.py --model models/transfer_best_model.keras --data fer2013
```

### Compare two models

```bash
python test_model.py --model models/best_model.keras --compare_model models/transfer_best_model.keras --data fer2013
```

### Architecture-only checks

These are useful when you want to verify that the model builds and runs without needing a trained checkpoint:

```bash
python test_model.py --arch_only
python test_model.py --arch_only --transfer_arch_only
```

### Optional single-image prediction

```bash
python test_model.py --model models/best_model.keras --data fer2013 --image path/to/face.jpg
```

### Testing hyperparameters and CLI options

| Argument | Type | Default | Meaning | Practical effect |
|---|---:|---:|---|---|
| `--model` | path | `models/best_model.keras` | Primary model to evaluate | Must point to a trained `.keras` file |
| `--compare_model` | path | `None` | Optional second model for comparison | Generates side-by-side comparison output |
| `--data` | path | `fer2013` | Dataset root | Required for full evaluation unless using architecture-only mode |
| `--arch_only` | flag | off | Skip loading a trained model and just build the baseline architecture | Good for sanity checks |
| `--transfer_arch_only` | flag | off | Build the MobileNetV2 transfer architecture in arch-only mode | Use together with `--arch_only` |
| `--image` | path | `None` | Predict a single image after evaluation | Uses the model’s saved metadata to preprocess correctly |

### Evaluation metrics produced

The script prints and saves:

- top-1 accuracy,
- top-2 accuracy,
- top-3 accuracy,
- classification report,
- confusion matrix,
- per-class precision / recall / F1,
- inference-speed benchmark, and
- optional single-image prediction output.

### Evaluation outputs

For the primary model:

- `logs/primary_model_confusion_matrix.png`
- `logs/primary_model_per_class_metrics.png`
- `logs/primary_model_classification_report.txt`

If `--compare_model` is used, the comparison summary is written to:

- `logs/model_comparison.json`

---

## 6) Web application (`app.py`)

The Flask app serves browser-based prediction for both model types.
It reads the `.metadata.json` sidecar file next to each `.keras` checkpoint, so preprocessing is automatic.

### Start the app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

### What the app supports

- model selection between baseline and transfer-learning checkpoints,
- face-image upload,
- webcam-style camera frame uploads from the browser,
- annotated results with emotion labels and probabilities.

### Browser note

The app starts with HTTPS enabled by default for camera compatibility in modern browsers.
If you want plain HTTP only, set:

```bash
export HER_DL_HTTPS=0
```

---

## 7) Recommended workflow

1. Train the baseline CNN.
2. Train the transfer-learning model.
3. Evaluate both with `test_model.py`.
4. Compare the confusion matrices and per-class metrics.
5. Run the Flask app with the better-performing checkpoint.

---

## 8) Troubleshooting

### `pip install -r requirements.txt` fails

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset folder not found

Make sure `fer2013/` contains both `train/` and `test/` subfolders.

### Training ends too early

- Increase `--epochs`.
- Reduce `--lr` if validation loss is unstable.
- For transfer learning, try a smaller `--fine_tune_lr`.
- Check whether early stopping is reaching its patience limit.

### Baseline overfits quickly

Try:

- `--include_brightness_augment`
- `--label_smoothing 0.1`
- `--no_class_weights` only if class imbalance handling is causing instability
- a smaller `--lr`

### Transfer model is unstable during fine-tuning

Try:

- a more negative `--fine_tune_at`,
- keeping `--freeze_bn` enabled,
- lowering `--fine_tune_lr`, and
- using fewer fine-tuning epochs.

### TensorFlow does not see the GPU

TensorFlow will fall back to CPU automatically. If you need GPU acceleration, check your CUDA, cuDNN, or ROCm installation.

---

## 9) Output summary

### Baseline run

- `models/best_model.keras`
- `models/best_model.weights.h5`
- `models/best_model.metadata.json`
- `logs/training_curves.png`
- `logs/training_history.csv`
- `logs/training_metadata.json`
- `logs/emotion_model_savedmodel/`

### Transfer-learning run

- `models/transfer_best_model.keras`
- `models/transfer_best_model.metadata.json`
- `logs/transfer/training_curves.png`
- `logs/transfer/training_metadata.json`
- `logs/transfer/stage1/history.csv`
- `logs/transfer/stage2/history.csv`
- `logs/transfer/savedmodel/`

### Evaluation run

- `logs/primary_model_confusion_matrix.png`
- `logs/primary_model_per_class_metrics.png`
- `logs/primary_model_classification_report.txt`
- `logs/model_comparison.json` when comparing two models
