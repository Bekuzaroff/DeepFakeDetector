# 🔍 Enhanced Deepfake Detection System with Ensemble Models


## 📁 Project Structure

```

.
├── prepare\_cifar\_binary.py          # Converts CIFAR-10 to binary real/fake dataset
├── multi\_model\_inference.py        # Training, evaluation, and inference pipeline
├── cifar\_binary/                   # Generated dataset directory (created automatically)
├── enhanced\_deepfake\_ensemble.pth  # Saved ensemble model (after training)
├── evaluation\_results/             # Metrics, plots, and reports
└── sample\_image.jpg                # Optional test image for demo inference

````

---

## ✅ Features

- 🔄 **Ensemble of models:** EfficientNet, ResNet, Vision Transformer
- 🧠 **Face extraction option:** Built-in OpenCV face detection support
- ⚖️ **Handles class imbalance:** Weighted sampling during training
- 📊 **Detailed evaluation:** AUC, ROC, Precision-Recall, Confusion Matrix
- 🧪 **Single image inference:** Predicts with ensemble and confidence score
- 🗂️ **Synthetic dataset preparation:** From CIFAR-10 (real/fake classes)

---

## ⚙️ Setup

### 🔧 Install Dependencies

```bash
pip install torch torchvision timm transformers scikit-learn matplotlib seaborn tqdm opencv-python
````

---

## 📦 Step 1: Prepare Dataset from CIFAR-10

1. Download [CIFAR-10 Python version](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Update the `cifar_dir` variable in `prepare_cifar_binary.py` with the path to your dataset.
3. Run:

```bash
python prepare_cifar_binary.py
```

This creates the dataset in the following structure:

```
cifar_binary/
├── train/
│   ├── real/
│   └── fake/
├── val/
├── test/
```

### Label Mapping:

* **Real (Label 0):** cat, deer, dog, horse
* **Fake (Label 1):** airplane, automobile, ship, truck

---

## 🚀 Step 2: Train the Deepfake Detection Ensemble

```bash
python multi_model_inference.py
```

This will:

* Train EfficientNet, ResNet, and ViT models
* Save best weights per model (e.g., `best_resnet50.pth`)
* Save the final ensemble as `enhanced_deepfake_ensemble.pth`
* Generate evaluation plots and metrics in `evaluation_results/`

---

## 🔍 Step 3: Inference on a Sample Image (Optional)

Place a file named `sample_image.jpg` in the project root and re-run the script:

```bash
python multi_model_inference.py
```

You will see per-model and ensemble predictions with confidence scores.

---

## 📈 Evaluation Output

* `evaluation_results/evaluation_report.json`: Accuracy, AUC, classification report
* `evaluation_results/evaluation_plots.png`: Confusion matrix, ROC, PR curves

---

## 💡 Notes

* Face extraction is optional and handled via OpenCV's Haar cascade.
* Early stopping and learning rate scheduling are included.
* The ensemble uses a **weighted average** strategy by default.

---

## 📜 License

For educational and research purposes only.
