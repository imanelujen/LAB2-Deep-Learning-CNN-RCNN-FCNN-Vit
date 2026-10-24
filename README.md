# Deep Learning Lab 2 — Computer Vision with CNN, Faster R-CNN, VGG16, AlexNet and Vision Transformer

This repository contains the implementation of **Deep Learning Lab 2**, focused on computer vision using PyTorch. The experiments use the MNIST handwritten digit dataset to explore image classification, object detection, transfer learning, and Vision Transformers (ViT).

## Notebook

The complete implementation is available in the following Jupyter notebook:

**[Open the notebook — CNN, RCNN, FCNN and ViT for Computer Vision](https://github.com/imanelujen/LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit/blob/main/cnn-rcnn-fcnn-for-computer-vision.ipynb)**

The experiments were conducted using Kaggle Notebooks with GPU acceleration.

## Objectives

- Implement and train a custom CNN for handwritten digit classification.
- Explore Faster R-CNN for object detection on MNIST images.
- Fine-tune pretrained VGG16 and AlexNet models for digit classification.
- Implement a Vision Transformer (ViT) from scratch.
- Compare model performance using accuracy, weighted F1-score, loss, and training time.
- Analyze the advantages and limitations of convolutional and Transformer-based architectures.

## Dataset

**MNIST** is a dataset of grayscale images representing handwritten digits from 0 to 9.

| Property | Description |
|---|---|
| Number of classes | 10 |
| Image dimensions | 28 × 28 pixels |
| Original training set | 60,000 images |
| Test set | 10,000 images |
| Training subset | 54,000 images |
| Validation subset | 6,000 images |

The original training set was divided into training and validation subsets. The test set was kept separate for the final evaluation of the classification models.

For VGG16 and AlexNet, grayscale images were converted to three channels, resized to 128 × 128 pixels, and normalized using ImageNet statistics.

For Faster R-CNN, the images were adapted to the object detection format by constructing bounding boxes around the foreground digit regions.

## Models and Methodology

### 1. Custom CNN

A convolutional neural network was implemented using PyTorch.

The architecture includes:

- Two convolutional layers with ReLU activation.
- Max-pooling for spatial downsampling.
- Fully connected layers for classification.
- Dropout for regularization.
- An output layer with ten classes.

The model was trained using the Adam optimizer, Cross-Entropy Loss, and a learning rate of 0.001. Weight decay was also used for regularization.

### 2. Faster R-CNN

Faster R-CNN with a pretrained ResNet-50 FPN backbone was adapted for MNIST object detection.

The original detection head was replaced to support ten digit classes plus the background class. The model was trained on a subset of 6,000 training images and evaluated on a validation subset of 1,000 images.

The highest-scoring detection was used to derive a digit prediction for the classification-style evaluation.

**Evaluation limitation:** Accuracy and F1-score based on the predicted digit class do not fully measure object detection performance. A more complete evaluation would include bounding-box localization metrics, such as Intersection over Union (IoU), and Average Precision (AP).

### 3. VGG16 Transfer Learning

VGG16 was initialized with pretrained ImageNet weights.

The original classification head was replaced with a classifier for ten digit classes. Most convolutional layers were frozen, while the final convolutional block and the new classification head were trained.

The model was fine-tuned using AdamW with a learning rate of 0.0001 and weight decay of 0.0001.

### 4. AlexNet Transfer Learning

AlexNet was initialized with pretrained ImageNet weights and adapted to classify MNIST digits.

Most convolutional layers were frozen, while later convolutional layers and a new classification head were trained.

The model was fine-tuned using AdamW with a learning rate of 0.0001 and weight decay of 0.0001.

### 5. Vision Transformer (ViT)

A Vision Transformer was implemented from scratch without pretrained weights.

The architecture includes:

- Patch embedding to divide each image into smaller patches.
- A learnable classification token.
- Positional embeddings to preserve patch position information.
- Transformer encoder layers with multi-head self-attention.
- A classification head with ten output classes.

For the MNIST images, a patch size of 7 × 7 was used, producing 16 patches per image. The Transformer uses an embedding dimension of 128, four attention heads, and four encoder layers.

Unlike CNNs, which learn local spatial features through convolutional filters, the ViT uses self-attention to model relationships between image patches.

## Experimental Results

The following results were obtained during the experiments.

| Model | Evaluation Accuracy | Weighted F1-score | Evaluation Loss | Training Time |
|---|---:|---:|---:|---:|
| Custom CNN | 99.07% | 0.9907 | 0.0272 | Not recorded |
| VGG16 | 99.29% | 0.9929 | 0.0320 | 617.1 s |
| AlexNet | 99.10% | 0.9910 | 0.0326 | 152.7 s |
| Faster R-CNN | 98.10%* | 0.9809* | Not directly comparable | 877.7 s |
| Vision Transformer | 97.49% | 0.9749 | 0.0870 | 67.7 s |

**Evaluation notes:**

- CNN, VGG16, and AlexNet were evaluated on the 10,000-image MNIST test set.
- The ViT results are reported as obtained in the experiment. Refer to the notebook for the exact evaluation split and training configuration.
- Faster R-CNN was evaluated on a validation subset of 1,000 images. Its results are not directly comparable to those of the classification models evaluated on the test set.
- Faster R-CNN's accuracy and F1-score are based on the class of the highest-scoring detection, not a complete object detection evaluation.
- The reported training times depend on the execution environment, GPU configuration, and experimental settings.
- The CNN training time was not recorded in the results summarized here.

## Results Analysis

### Custom CNN

The custom CNN achieved **99.07% accuracy** and a weighted F1-score of 0.9907 on the MNIST test set. This shows that a relatively compact convolutional architecture can achieve excellent performance on handwritten digit classification.

### VGG16

VGG16 achieved the highest test accuracy among the three classification models evaluated on the 10,000-image test set, reaching **99.29%**.

Its training time was approximately 10 minutes and 17 seconds for three epochs. Although it achieved the best accuracy in this comparison, it required considerably more training time than AlexNet.

### AlexNet

AlexNet achieved **99.10% test accuracy** with a training time of approximately 2 minutes and 33 seconds for three epochs.

Its performance was close to that of the custom CNN and VGG16, while requiring substantially less training time than VGG16 in this experiment.

### Faster R-CNN

Faster R-CNN achieved 98.10% accuracy and a weighted F1-score of 0.9809 on the selected validation subset.

However, Faster R-CNN is designed for object detection rather than image classification alone. A more meaningful evaluation of its detection capabilities would measure both the correctness of the predicted digit classes and the accuracy of the predicted bounding boxes.

Its longer training time also reflects the additional computational work involved in object detection.

### Vision Transformer

The ViT achieved **97.49% accuracy**, a weighted F1-score of 0.9749, and an evaluation loss of 0.0870, with a reported training time of 67.7 seconds.

Its accuracy was lower than that of the evaluated CNN-based classification models. This is consistent with the fact that a ViT trained from scratch must learn useful visual representations without pretrained weights. The result also depends on the model configuration, training duration, and dataset.

The experiment illustrates an important difference between CNNs and Transformers: CNNs use convolutional filters to extract local spatial patterns, whereas ViTs use self-attention to model relationships between image patches.

The ViT's shorter reported training time should be interpreted in the context of its specific configuration and experimental setup; it does not establish that Transformers are generally faster than CNNs.

### Overall Comparison

Among the classification models evaluated on the MNIST test set, VGG16 obtained the highest accuracy, followed closely by AlexNet and the custom CNN. AlexNet offered a useful balance between accuracy and training time.

The ViT provided a Transformer-based comparison, while Faster R-CNN introduced the additional task of localizing objects within images. Because the evaluation datasets and metrics differ for Faster R-CNN, its results should be interpreted separately.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- MNIST dataset
- Kaggle Notebooks
- CUDA GPU acceleration

A GPU is recommended, particularly for Faster R-CNN and the transfer-learning experiments.

## How to Run

1. Open the notebook using the GitHub link above.
2. Import or load the notebook into Kaggle.
3. Enable GPU acceleration in the notebook settings.
4. Run the cells in order to prepare the dataset, initialize the models, train them, and evaluate their performance.
5. Review the generated metrics and plots.

The exact package versions and execution settings used for each experiment can be checked in the notebook.

## Conclusion

This lab explores several approaches to computer vision through handwritten digit classification and object detection.

The custom CNN provides a strong baseline with high accuracy and a relatively compact architecture. Transfer learning with VGG16 and AlexNet also achieves excellent results, with different training costs. Faster R-CNN demonstrates how object detection extends classification by predicting both object classes and bounding boxes. The ViT experiment explores a Transformer-based approach to visual representation learning.

The results highlight the importance of considering not only accuracy and F1-score, but also training time, model architecture, evaluation methodology, and the task each model is designed to solve.

## Repository

[LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit](https://github.com/imanelujen/LAB2-Deep-Learning-CNN-RCNN-FCNN-Vit)
