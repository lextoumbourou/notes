---
name: evaluate-image-classifier
description: Evaluate a trained image classifier on held-out images. Use when asked for accuracy or example predictions.
---

# Evaluate an image classifier

1. Find the dataset's official test split. Load the saved model and use the same preprocessing as training.
2. Run inference on the test images without training on them. Count correct predictions and total images.
3. Show a few test images with their predicted and true labels, including some mistakes.
4. Report the correct count, total count, accuracy, test split, and whether the model used pretrained weights.
5. Keep model selection and tuning on training or validation data. Do not choose a model using the test labels.
