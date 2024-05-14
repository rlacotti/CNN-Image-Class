# CNN Image Classification Project

## Description
The CNN Image Classification Project is a Python project aimed at implementing a Convolutional Neural Network (CNN) for image classification tasks. The project utilizes the PyTorch framework to build and train the CNN model, which is then evaluated on a dataset consisting of images of various characters from the popular anime *One Piece*. By leveraging CNNs, the project seeks to accurately classify images into predefined character categories, showcasing the potential of deep learning techniques in image recognition tasks.

## Features
* **PyTorch Integration**: Utilizes the PyTorch framework for building, training, and evaluation of the CNN Model.
* **Data Preprocessing**: Implements sophisticated data preprocessing techniques such as normalization and data augmentation to enhance model performance and robustness.
* **Evaluation metrics**: Evaluates model performance using a range of metrics including accuracy, precision, recall, and loss, providing insights into the model's efficacy.
* **Visualizations**: Generates dynamic visulaizations of training and testing metrics, enabling intuitive assessment of model performance over epochs.
* **Checkpoint Management**: Provides functionality to save and load model checkpoints, facilitating seamless continuation of training or inference. Additionally, maintains a metrics log text file, capturing detailed performance metrics from each individual epoch such as: accuracies, losses, precision, and recall for comprehensive analysis and comparison.

## Dataset
This project utilizes a curated datset comprising of images of the Strawhat Pirates from *One Piece*. Organized into strutured subdirectories, each representing a distinct character category, this dataset serves as the cornerstone for training and evaluating the CNN model. With a focus on data integrity and relevance, the dataset ensures the robustness and accuracy of the image classification tasks undertaken by the project.

 ### Subdirectories
  * Brook
  * Chopper
  * Franky
  * Jinbei
  * Luffy
  * Nami
  * Robin
  * Sanji
  * Usopp
  * Zoro
 
 ## Repository Structure
 * **checkpoints**: Stores saved model checkpoints for future use, along with metrics log text file.
 * **DATA**: Contains the subdirectories, which in turn, house the images used for training and testing.
 * **Plots**: Houses visualizations of training and testing metrics.
 * **CNN.py**:  Source code file for the Convolutional Neural Network (CNN) model implementation, showcasing deep learning techniques for image classification tasks.
 * **Eval.py**: File for evaluating the CNN model, providing insights into its performance metrics.
 * **Loaddata.py**: File for loading and preprocessing the dataset, ensuring compatibility with the CNN model training process.

  
