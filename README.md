# Data science portfolio by Douglas Woollam

This portfolio is a collection of notebooks I made for data analysis or machine learning algorithm exploration.

## Stand-alone projects.

---

## Handwritten Digit Recognition with MNIST

[Github](https://github.com/AirDoogle/data_science_portfolio/blob/main/ml_handwritten_digit_recognition.ipynb) [nbviewer](https://nbviewer.org/github/AirDoogle/data_science_portfolio/blob/main/ml_handwritten_digit_recognition.ipynb)

This project demonstrates the implementation of a machine learning model to classify handwritten digits using the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). The notebook utilizes TensorFlow and Keras for training a neural network on the dataset. Key tasks include:

- Loading and preprocessing the MNIST dataset.
- Building and training a neural network model.
- Evaluating the model's performance on unseen data.

This project highlights foundational deep learning techniques, such as data normalization, model training, and evaluation metrics, to solve a classic problem in computer vision.

--- 

## Classification problems.

### Titanic: Machine Learning from Disaster

[Github](https://github.com/AirDoogle/data_science_portfolio/blob/main/ml_titanic__knn_nn.ipynb) [nbviewer](https://nbviewer.org/github/AirDoogle/data_science_portfolio/blob/main/ml_titanic__knn_nn.ipynb)

Titanic: Machine Learning from Disaster is a knowledge competition on Kaggle. Many people started practicing in machine learning with this competition, so did I. This is a binary classification problem: based on information about Titanic passengers we predict whether they survived or not. General description and data are available on [Kaggle](https://www.kaggle.com/c/titanic).
Titanic dataset provides interesting opportunities for feature engineering.

### Image Classification with CIFAR-10

[GitHub](https://github.com/AirDoogle/data_science_portfolio/blob/main/ml_cifar-10_image_categorisation.ipynb) | [nbviewer](https://nbviewer.org/github/AirDoogle/data_science_portfolio/blob/main/ml_cifar-10_image_categorisation.ipynb)

The CIFAR-10 dataset is a widely used benchmark in computer vision for image classification tasks. In this project, I applied deep learning techniques to classify images from the CIFAR-10 dataset. This is a multi-class classification problem: based on image data, to predict which of the ten categories the image belongs to, such as airplanes, cars, birds etc. The dataset and description are available on the [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html). Working with CIFAR-10 allowed me to explore convolutional neural networks (CNNs), data augmentation, and techniques to improve model generalization.

<!--This project applies deep learning techniques to classify images from the CIFAR-10 dataset, consisting of 60,000 images across 10 categories. The goal is to develop a robust image classification model using convolutional neural networks (CNNs). Key tasks include:  

- **Data Preprocessing and Augmentation**: Normalizing pixel values, applying one-hot encoding, and using data augmentation (random flipping and rotation) to improve generalization.  
- **Model Development**: Implementing multiple convolutional architectures, including variations with and without pooling layers, batch normalization, and dropout to evaluate trade-offs in accuracy and training time.  
- **Optimization Techniques**: Using hyperparameter tuning, Adam optimizer, and categorical cross-entropy loss to enhance model performance.  
- **Evaluation and Trade-offs**: Comparing different model architectures, highlighting trade-offs between computational time and accuracy, and achieving a final test accuracy of **76.24%** with a balanced model structure.  

This project demonstrates key deep learning concepts such as feature extraction, regularization, and performance evaluation in image classification.-->

### Bamboo Forest Detection from Satellite Images

[Github](https://github.com/AirDoogle/data_science_portfolio/blob/main/ml_bamboo_forest_detection.ipynb) [nbviewer](https://nbviewer.org/github/AirDoogle/data_science_portfolio/blob/main/ml_bamboo_forest_detection.ipynb)

This project applies deep learning techniques to detect bamboo forests in satellite imagery. Using convolutional neural networks (CNNs), the model classifies image patches as containing bamboo or not. Key tasks include:  

- **Data Preprocessing and Augmentation**: Segmenting large satellite images into smaller patches using a sliding window approach, normalizing pixel values, and handling class imbalances with computed weights.  
- **Model Development**: Implementing a CNN with convolutional layers, max pooling, dropout regularization, and batch normalization for robust classification.  
- **Optimization and Evaluation**: Using binary cross-entropy loss, Adam optimizer, and accuracy metrics to assess model performance. Final model achieves strong classification accuracy and generalizes well.  
- **Prediction and Visualization**: Generating a probability heatmap over satellite images to highlight bamboo forest regions, scaling and overlaying predictions for interpretability.  

This project demonstrates advanced computer vision techniques in remote sensing, showcasing how deep learning can assist in environmental monitoring and land classification.  


### Algorithm Selection for Model Counting (#SAT)

[Github](https://github.com/AirDoogle/data_science_portfolio/blob/main/ml_sat_algorithm_selection.ipynb) [nbviewer](https://nbviewer.org/github/AirDoogle/data_science_portfolio/blob/main/ml_sat_algorithm_selection.ipynb)

This project explores the use of machine learning techniques for algorithm selection in the #SAT problem, an extension of Boolean satisfiability. The primary objective is to classify the optimal solver for each instance based on a set of 72 features derived from the problem. Key tasks include:

- **Preprocessing and Feature Selection**: Leveraging Recursive Feature Elimination (RFE) and Principal Component Analysis (PCA) for dimensionality reduction and feature optimization.  
- **Model Training and Evaluation**: Training a k-Nearest Neighbors (k-NN) classifier, Decision Tree, and Random Forest models to identify the optimal solver for each problem instance.  
- **Optimization Techniques**: Employing cross-validation, hyperparameter tuning, and feature normalization to enhance model performance and reliability.

This project demonstrates foundational techniques in data preprocessing, model evaluation, and algorithmic optimization for solving NP-complete problems.

<!-- ## Regression problems.

### House Prices: Advanced Regression Techniques

House Prices: Advanced Regression Techniques is a knowledge competition on Kaggle. This is a regression problem: based on information about houses we predict their prices. General description and data are available on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
The dataset has a lot of features and many missing values. This gives interesting possibilities for feature transformation and data visualization.


## Natural language processing.

### Bag of Words Meets Bags of Popcorn


Bag of Words Meets Bags of Popcorn is a sentimental analysis problem. Based on texts of reviews we predict whether they are positive or negative. General description and data
The data provided consists of raw reviews and class (1 or 2), so the main part is cleaning the texts.

### NLP with Python: exploring Fate/Zero

Natural language processing in machine learning helps to accomplish a variety of tasks, one of which is extracting information from texts. This notebook is an overview of several text exploration methods using English translation of Japanese light novel "Fate/Zero" as an example.

### NLP. Text generation with Markov chains


This notebook shows how a new text can be generated based on a given corpus using an idea of Markov chains. I start with simple first-order chains and with each step improve model to generate better text.

### NLP. Text summarization


This notebook shows how text can be summarized choosing several most important sentences from the text. I explore various methods of doing this based on a news article.

## Clustering

### Clustering with KMeans


Clustering is an approach to unsupervised machine learning. Clustering with KMeans is one of algorithms of clustering. in this notebook I'll demonstrate how it works. Data used is about various types of seeds and their parameters. It is availae
## Neural networks

### Feedforward neural network with regularization


This is a simple example of feedforward neural network with regularization. It is based on Andrew Ng's lectures on Coursera. I used data from Kaggle's challenge "Ghouls, Goblins, and Ghosts... Boo!".

## Data exploration and analysis

### Telematic data

I have a dataset with telematic information about 10 cars driving during one day. I visualise data, search for insights and analyse the behavior of each driver. I can't share the data, but here is the notebook. I want to notice that folium map can't be rendered by native github, but nbviewer.jupyter can do it.

## Recommendation systems.

### Collaborative filtering


Recommenders are systems, which predict ratings of users for items. There are several approaches to build such systems and one of them is Collaborative Filtering. 
This notebook shows several examples of collaborative filtering algorithms.
-->
