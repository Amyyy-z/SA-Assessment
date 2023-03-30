# SA-Assessment
This repository contained the overall implementation procedures for the brain electrical activity mapping (BEAM) images-based three-level situation awareness (SA) assessment.

## Getting Started 

Before implementing the classification tasks, there are some pre-requisites to comply:

- Computational environment: Tensorflow 2.6.0, Python 3.7 at least, and GPU encouraged
- Data pre-processing completed:
  * All the images must be in the same size, in this project, we have an input size of 224*224*3
  * Data augmentation is encouraged if class imbalance issue happens

## Implementation Guide

* Prepare the image sets 
* Store the image sets into different folders based on their class
* Assign labels to each folder for each class
* Import required libraries
* Import image sets with their labels
* Encode image labels with one-hot-encoding technique
* Training and Testing splits with training and validation, cross-validation if optional
* Construct individual learner CNN models (VGG, ResNet, Inception, Xception, DenseNet)
* Feed the images and encoded labels into the model for training and fine-tuning
* Evaluate a set of CNN models for comparison
* Select the best-performing model based on their performance (accuracy, auc, precision, recall, specificity, npv, f1, fpr)

---------------------------

### Enjoy!
