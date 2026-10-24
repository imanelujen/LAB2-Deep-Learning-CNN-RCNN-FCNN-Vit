# Deep Learning Lab 2 — Computer Vision with CNN, Faster R-CNN, VGG16, AlexNet and Vision Transformer

This repository contains the implementation of **Deep Learning Lab 2**, focused on computer vision using PyTorch. The experiments use the MNIST handwritten digit dataset to explore image classification, object detection, transfer learning, and Vision Transformers.

## Notebook

The complete implementation is available in the Jupyter notebook:

**[Open the notebook — CNN, RCNN, FCNN and ViT for Computer Vision](https://github.com/imanelujen/LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit/blob/main/cnn-rcnn-fcnn-for-computer-vision.ipynb)**

The experiments were conducted using Kaggle with GPU acceleration.

## Objectives

- Implement and train a CNN for handwritten digit classification.
- Explore Faster R-CNN for object detection on MNIST images.
- Fine-tune pretrained VGG16 and AlexNet models for digit classification.
- Build a Vision Transformer (ViT) from scratch.
- Compare model performance using accuracy, weighted F1-score, loss, and training time.
- Analyze the advantages and limitations of convolutional and Transformer-based architectures.

## Dataset

**MNIST** contains grayscale images of handwritten digits from 0 to 9.

| Property | Description |
|---|---|
| Number of classes | 10 |
| Image dimensions | 28 × 28 pixels |
| Number of training images | 60,000 |
| Number of test images | 10,000 |
| Training subset | 54,000 images |
| Validation subset | 6,000 images |

The original training set was divided into training and validation subsets. The test set was kept separate for final evaluation.

For VGG16 and AlexNet, images were converted to three channels, resized to 128 × 128, and normalized using ImageNet statistics.

For Faster R-CNN, the MNIST images were adapted to the object detection format by constructing bounding boxes around the foreground digit regions.

## Models and Methodology

### 1. Custom CNN

A convolutional neural network was implemented using PyTorch.

The architecture includes:
- Convolutional layers with ReLU activation.
- Max-pooling layers for spatial downsampling.
- Fully connected layers for classification.
- Dropout for regularization.

The model was trained using the Adam optimizer and Cross-Entropy Loss.

### 2. Faster R-CNN

Faster R-CNN with a pretrained ResNet-50 FPN backbone was adapted for MNIST object detection.

The original detection head was replaced to support the ten digit classes and the background class. The model was trained on a subset of the MNIST training data.

The highest-scoring detection was used to derive a digit prediction for the classification-style evaluation. These accuracy and F1 results do not measure bounding-box localization quality; metrics such as Intersection over Union (IoU) and Average Precision (AP) would be needed for a more complete detection evaluation.

### 3. VGG16 Transfer Learning

VGG16 was initialized with pretrained ImageNet weights.

The original classification head was replaced with a new classifier for ten digit classes. Most convolutional layers were frozen, while the final convolutional block and the new classification head were trained.

### 4. AlexNet Transfer Learning

AlexNet was initialized with pretrained ImageNet weights and adapted to classify MNIST digits.

Most convolutional layers were frozen, while the later convolutional layers and a new classification head were trained.

### 5. Vision Transformer (ViT)

The ViT implementation is designed to be built from scratch, without pretrained weights.

The architecture uses:
- Patch embedding to divide each image into smaller patches.
- A learnable classification token.
- Positional embeddings.
- Transformer encoder layers with multi-head self-attention.
- A classification head with ten output classes.

The ViT results will be added after the training and test evaluation are completed.

## Experimental Results

The following results were obtained during the experiments.

| Model | Test / Evaluation Accuracy | Weighted F1-score | Loss | Training Time |
|---|---:|---:|---:|---:|
| Custom CNN | 99.07% | 0.9907 | 0.0272 | Not recorded here |
| VGG16 | 99.29% | 0.9929 | 0.0320 | 617.1 s |
| AlexNet | 99.10% | 0.9910 | 0.0326 | 152.7 s |
| Faster R-CNN | 98.10%* | 0.9809* | Not directly comparable | 877.7 s |
| Vision Transformer | To be completed | To be completed | To be completed | To be completed |

**Evaluation notes:**
- CNN, VGG16, and AlexNet were evaluated on the 10,000-image MNIST test set.
- Faster R-CNN was evaluated on a validation subset of 1,000 images. Its results are therefore not directly comparable to the test results of the classification models.
- The Faster R-CNN evaluation used the class of the highest-scoring detection, rather than a full object detection metric.
- Training times depend on the execution environment and GPU configuration.

## Results Analysis

### CNN

The custom CNN achieved **99.07% test accuracy** and a weighted F1-score of 0.9907. This demonstrates that a relatively compact convolutional architecture can perform very well on MNIST.

### VGG16

VGG16 achieved the highest test accuracy among the three evaluated classification models, reaching **99.29%**. Its training time was approximately 10 minutes and 17 seconds for three epochs.

### AlexNet

AlexNet achieved **99.10% test accuracy** with a training time of approximately 2 minutes and 33 seconds for three epochs. Its performance was close to that of the custom CNN and VGG16, while requiring less training time than VGG16 in this experiment.

### Faster R-CNN

Faster R-CNN achieved 98.10% accuracy on the selected validation subset. Since it is an object detection model, a complete evaluation should also measure how accurately the predicted bounding boxes localize the digits.

### Vision Transformer

The ViT experiment is intended to examine how self-attention and patch embeddings can be applied to image classification. Its final performance and training cost will be compared with the CNN-based models once the experiment is complete.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- MNIST
- Kaggle Notebooks
- GPU acceleration with CUDA


A GPU is recommended, especially for Faster R-CNN, VGG16, and AlexNet.

## Conclusion

This lab explores several approaches to computer vision using a common handwritten digit classification task.

The custom CNN provides a strong baseline. Transfer learning with VGG16 and AlexNet achieves similarly high performance, with different training costs. Faster R-CNN introduces the additional challenge of object localization, while the ViT experiment explores a Transformer-based approach to visual representation learning.

The experiments highlight the importance of considering not only accuracy and F1-score, but also training time, model architecture, evaluation methodology, and the specific task each model is designed to solve.

[LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit](https://github.com/imanelujen/LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit)
