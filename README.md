# Forest Fire Detection Experiment

Binary classification experiment for detecting forest fires using three different models: EfficientNet, MobileNet, and SVM.

## Experiment Setup

### Dataset Structure
```
data/
├── train/
│   ├── fire/          # 600 fire images
│   └── non_fire/      # 600 forest non-fire images
├── test/
│   ├── fire/          # Part of 120 test images
│   └── non_fire/      # Part of 120 test images
└── bcst/              # 200 BCST (fire-like/sunlight) images
```

### Training Configuration
- **Baseline Training**: 600 fire images + 600 forest non-fire images
- **Test Set**: 120 images (fire + non-fire)
  - 200 BCST images added to negative class
  - 200 forest images removed from negative class

### Models
1. **EfficientNet-B0** (PyTorch, pre-trained)
2. **MobileNet-V2** (PyTorch, pre-trained)
3. **SVM** (scikit-learn, using ResNet50 features)

### Metrics
- **Fire Detection Rate** = TP / (TP + FN)
- **Error Warning Rate** = FP / (FP + TN)
- Confusion matrices for each model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your dataset according to the structure above.

## Usage

### Run Complete Experiment

Run the full experiment with all three models:

```bash
python experiment_one.py --data-root data --epochs 20 --lr 0.001 --patience 5
```
```bash
python experiment_color_space.py --data-root data --epochs 20 --lr 0.001 --patience 5
```

### Arguments
- `--data-root`: Root directory containing data folders (default: `data`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Output

The experiment generates:
- `results/comparison_table.csv`: Comparison table of all models
- `results/detailed_results.json`: Detailed metrics in JSON format
- `results/efficientnet_confusion_matrix.png`: EfficientNet confusion matrix
- `results/mobilenet_confusion_matrix.png`: MobileNet confusion matrix
- `results/svm_confusion_matrix.png`: SVM confusion matrix
- `checkpoints/`: Saved model checkpoints

## Project Structure

```
jingyao-fire-detection/
├── data/                  # Dataset (you need to organize your images here)
├── models/                # Model implementations
│   ├── efficientnet.py
│   ├── mobilenet.py
│   └── svm_model.py
├── utils/                 # Utility functions
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── metrics.py        # Custom metrics
│   └── visualization.py  # Plotting utilities
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── experiment.py         # Main experiment orchestrator
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Notes

- The code automatically uses GPU if available, otherwise falls back to CPU
- Training uses data augmentation (random flips, rotations, color jitter)
- Models are saved in `checkpoints/` directory
- Best model (based on validation accuracy) is automatically loaded for evaluation

