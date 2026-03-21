# Real-Time-Face-Detection

A production-ready face mask detection system using TensorFlow, Keras, and OpenCV for real-time classification of whether a person is wearing a mask or not.

## Project Overview

This project implements a deep learning-based face mask detector that can classify faces in images or real-time video streams. It uses transfer learning with MobileNetV2 as the base model, fine-tuned for binary classification (mask/no mask).

## Dataset Format

The dataset should be organized as follows:

```
dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each subdirectory contains images of faces with and without masks respectively.

## Installation

1. Clone or download the project.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

### Using the Notebook

1. Open `notebooks/training.ipynb` in Jupyter.
2. Run all cells to train the model.
3. The trained model will be saved to `models/mask_detector.h5`.

### Using the Script

For production training:

```bash
cd src
python train.py --dataset_path ../dataset
```

This will:
- Train the model with early stopping and model checkpointing.
- Print classification metrics and confusion matrix.
- Save the ROC curve plot.
- Save the best model automatically.

## Webcam Detection

To run real-time detection:

```bash
cd src
python detect.py
```

This will:
- Open your webcam.
- Detect faces using Haar cascades.
- Classify each face as "Mask" or "No Mask".
- Display bounding boxes (green for mask, red for no mask) with confidence scores.
- Show FPS counter.
- Press 'q' to quit.

## Example Output

During detection, you'll see:
- Green boxes around faces with masks.
- Red boxes around faces without masks.
- Confidence scores displayed above each box.
- Real-time FPS in the top-left corner.

## Performance

The model achieves high accuracy on the validation set with transfer learning. Typical performance:
- Accuracy: ~95-98%
- Precision/Recall: Balanced for both classes
- Real-time inference: 20-30 FPS on modern hardware

For best results:
- Use a dataset with diverse lighting conditions and angles.
- Ensure faces are clearly visible in images.
- Fine-tune hyperparameters if needed.

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Imutils

## Notes

- The Haar cascade classifier is used for face detection; it may not detect faces in all orientations.
- Model expects 224x224 RGB images.
- Ensure your webcam is properly connected for detection.
