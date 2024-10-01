# Impact of Pre-training Datasets on Human Activity Recognition with Contrastive Predictive Coding.

This repo contains the PyTorch implementation of the paper: [Contrastive Predictive Coding for Human Activity Recognition](https://dl.acm.org/doi/10.1145/3463506), accepted at IMWUT 2021.  
Some `.py` files have been modified to support a new experimental scenario for the paper: **Impact of Pre-training Datasets on Human Activity Recognition with Contrastive Predictive Coding**.

## Introduction

Contrastive Predictive Coding (CPC) is a self-supervised learning (SSL) technique that has demonstrated promising results in several tasks, including human activity recognition (HAR). In this work, we explore the impact of data variety on backbone pre-training when designing CPC models for HAR and the benefits of pre-training on the final model.

## Overview

The Activity Recognition Chain (ARC) comprises five distinct steps: data collection, pre-processing, windowing, feature extraction, and classification.  
In this work, we focus on the fourth step—feature extraction—and accomplish it via self-supervision.

We first split the dataset by participants to form the `train-val-test` splits. Of the available participants, 20% are randomly chosen for testing, while the remaining 80% are further divided randomly into the train and validation splits at an 80:20 ratio. The training data is normalized to have zero mean and unit variance, which is subsequently applied to the validation and test splits as well.

The sliding window process is applied to segment one second of data with 50% overlap. On these unlabeled windows, we first pre-train the CPC model, and transfer the learned encoder weights to a classifier for activity recognition (or fine-tuning). During classification, the learned encoder weights are frozen, and only the MLP classifier is updated using cross-entropy loss with the ground truth labels.

Additionally, we evaluated the impact of data variety on model pre-training using 15 combinations of four distinct HAR datasets, finding significant performance variability based on the pre-training datasets.

## Training

In this repo, pre-training can be performed using the following command:

```bash
python main.py --dataset UCI_Raw
```

The trained model weights are saved to the `models/<DATE>` folder, where `<DATE>` corresponds to the date the training was completed. By specifying the dataset as `UCI_Raw`, the processed data is utilized based on the location specified in `arguments.py`.  
The pre-training logs are saved in the `saved_logs/<DATE>` folder and contain the losses as well as the accuracy of the CPC pre-training.

## Activity Recognition/Classification

Once the model training is complete, the saved weights can be used for classification by running:

```bash
python evaluate_with_classifier.py --dataset UCI_Raw --saved_model <FULL_PATH_TO_MODEL.pkl>
```

The classification logs are saved in the `saved_logs/<DATE>` folder and contain the losses, accuracy, and F1-scores for the classification.

Another option is to use:

```bash
python run_experiments.py --saved_model <FULL_PATH_TO_MODEL.pkl>
```

This command runs `python evaluate_with_classifier.py` for all four target datasets (KuHar, MotionSense, UCI, and RealWorld_waist) with different percentages of target dataset data, outputs the results to a `.csv` file, and plots the curve.

## Summary of Files in the Repository

| File name                   | Function                                                                                                                        |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `arguments.py`               | Contains the arguments used for pre-training and classification.                                                                |
| `dataset.py`                 |Dataset and data loaders to load the pre-processed data and apply the sliding window procedure.                                  |
| `evaluate_with_classifier.py`| To train the classifier on the learned encoder weights.                                                                         |
| `main.py`                    | Over-arching file for the pre-training.                                                                                         |
| `meter.py`                  | For logging metrics and losses.                                                                                         |
| `model.py`                  | Defines the model and the CPC loss.                                                                                        |
| `run_experiments.py`         | Runs the classification experiments on multiple datasets with different percentages of target data and logs the results.        |
| `sliding_window.py`         | Efficiently performs the sliding window process on numpy (adapted from [this blog](http://www.johnvinyard.com/blog/?p=268)).|
| `trainer.py`                | Contains methods for training the CPC.                                                                                     |
| `utils.py`                  | Utility functions.                                                                        |

## License

This software is released under the GNU General Public License v3.0.

## Repositories Utilized in This Project

This project is based on CPC implementations from the following repositories:

- [Contrastive Predictive Coding for Human Activity Recognition](https://github.com/harkash/contrastive-predictive-coding-for-har)
- [Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)
- [emotion_detection_cpc](https://github.com/McHughes288/emotion_detection_cpc)

These repositories were instrumental in the development of this project.

