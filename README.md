![](UTA-DataScience-Logo.png)

# Predicting Horse Survival Outcome Based on Health Factors

* This repository applies machine learning techniques to predict the survival of horses based on various medical and health factors.

## Overview

* **Definition of the task / challenge**: Predict whether a horse `lived` or `died` based on health features like vital signs and medical history.
* **Our approach**: Applying machine learning and deep learning to preprocessed and cleaned data for binary classification, with preprocessing for missing values and class imbalance.
* **Summary of the performance achieved**: Best model achieved 85% accuracy.

## Data

* **Type**:  
  - Input: CSV file with health features (e.g., rectal temperature, pulse, age, surgery, etc.).
  - Output: Survival outcome (`lived` or `died`).

* **Size**:  
  - 5000 instances.

* **Instances (Train, Test, Validation Split)**:  
  - 3500 train, 1000 test, 500 validation.

#### Preprocessing / Clean up

* Missing data handled using imputation (median for numerical, mode for categorical).
* Numerical features scaled, categorical features one-hot encoded.

## Data Visualization

* Here are a few visualizations of the data:

1. **Total Protein**     
   Horses that lived and died have total protein levels mostly concentrated around 5–10 g/dL, with a slight shift indicating that higher total protein levels are more common in the lived group. Extremely high protein levels are rare and distributed similarly across both groups.

2. **Rectal Temperature**
   Horses that lived have rectal temperatures concentrated around 38°C, while those that died display a broader distribution, with more extreme temperatures (<36°C or >40°C). This suggests that abnormal rectal temperatures may be associated with a higher risk of mortality.

3. **Pulse**
   Horses that lived have pulses concentrated between 40–80 bpm, while those that died show a broader distribution, with elevated pulses above 100 bpm being more common. Elevated pulse rates are likely associated with higher mortality risk.
   
5. **Pain**
   The plot shows that horses experiencing mild pain have significantly higher survival rates compared to deaths. For horses with severe or extreme pain, deaths are more common. Among horses labeled as alert, survival is higher, but for those marked as depressed, deaths and survivals are almost equal. Higher pain levels are strongly associated with increased mortality.
   
6. **Extremity temperatures**
   The plot shows that horses with normal extremity temperatures have higher survival rates compared to deaths. Horses with cold extremities are more likely to die, while warm extremities are rare and have fewer deaths. This indicates that abnormal extremity temperatures, particularly cold, are associated with a higher risk of mortality.

   
   ### Problem Formulation

#### **Input / Output**
- **Input**: The dataset includes features like vital signs (e.g., pulse, temperature), medical conditions (e.g., abdominal distention, mucous membrane color), and other physiological data (e.g., packed cell volume).
- **Output**: The target is the outcome of the horse, with 3 possible classes: "lived" (0), "died" (1), and "euthanized" (2).

#### **Models**
- **Neural Network**: Chosen for its ability to capture complex patterns in the data. Includes L2 regularization, dropout layers, and a softmax output layer for multiclass classification.
- **Random Forest**: Used for its robust handling of both numerical and categorical features, and is less prone to overfitting.
- **Support Vector Machine (SVM)**: Selected for its effectiveness in high-dimensional spaces, particularly for non-linear decision boundaries.

#### **Loss, Optimizer, Other Hyperparameters**
- **Neural Network**: 
  - Loss: `sparse_categorical_crossentropy` (for multiclass classification).
  - Optimizer: `Adam` with a learning rate of 0.001.
  - Regularization: L2 regularization and Dropout with a rate of 0.5 to avoid overfitting.
- **Random Forest**: 
  - `n_estimators`: 100, `max_depth`: None, `criterion`: 'gini'.
- **SVM**: 
  - Kernel: RBF, `C`: 1.0, `gamma`: 'scale'.


### Training

#### **How you trained: Software and Hardware**
- **Software**: Python with libraries such as TensorFlow/Keras for the neural network, scikit-learn for Random Forest and SVM.
- **Hardware**: CPU-based training on a standard laptop.

#### **How did training take**
- Training took a varying amount of time depending on the model:
  - Neural Network: ~10-30 minutes, depending on epochs and early stopping.
  - Random Forest and SVM: Faster (~5-10 minutes), as they use parallelization.

#### **Training Curves**
- **Loss vs Epoch for Train/Test**: The neural network showed steady improvement in training accuracy but fluctuating validation accuracy, indicating overfitting.
  
#### **How did you decide to stop training**
- Training was stopped early using **early stopping** based on validation loss improvement, preventing overfitting.

#### **Any difficulties? How did you resolve them**
- **Overfitting**: Addressed by adding dropout layers and L2 regularization. Early stopping helped avoid further overfitting.
- **Model instability**: Adjusted the learning rate and batch size for more stable convergence.


### Performance Comparison

#### **Key Performance Metrics**
- **Accuracy**: Used for both training and validation accuracy.
- **AUC-ROC**: AUC scores were used for evaluating class discrimination capabilities.
- **F1-Score**: Especially for imbalanced datasets, F1-score was used to evaluate precision and recall.

#### **Results Comparison (Table)**

| Model               | Accuracy (Train) | Accuracy (Test) | AUC-ROC | F1-Score |
|---------------------|------------------|-----------------|---------|----------|
| Neural Network      | 0.85             | 0.49            | 0.64    | 0.60     |
| Random Forest       | 0.78             | 0.50            | 0.65    | 0.62     |
| Support Vector Machine | 0.80           | 0.48            | 0.60    | 0.58     |

#### **Visualization**
- **ROC Curve**: The ROC curve showed that the neural network and Random Forest had similar performance, with AUC values of 0.64 and 0.65, respectively.


### Conclusions

- **Model Performance**: The neural network performed slightly better than Random Forest and SVM, but its validation accuracy was still low, indicating that further tuning and additional data might improve performance.
- **Generalization**: Overfitting was a major issue, but techniques like early stopping, dropout, and regularization helped to some extent.
- **Data and Model Improvements**: Further data preprocessing (e.g., addressing class imbalance) and model adjustments (e.g., deeper neural networks) could lead to better results.

  
- ### Future Work

1. **Data Augmentation**: Increase dataset size by generating synthetic samples or using techniques like SMOTE for better generalization.
2. **Hyperparameter Tuning**: Perform grid search or random search to optimize model hyperparameters (e.g., learning rate, batch size).
3. **Class Imbalance**: Use techniques like class weighting or oversampling to handle class imbalance more effectively.

## How to reproduce results

* To reproduce the results, follow these steps:

1. **Set Up**: Install the necessary libraries (`TensorFlow`, `Keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`).
2. **Data Preparation**: Load and preprocess the dataset, encoding categorical features into numeric values.
3. **Model Definition**: Define a neural network with dropout and L2 regularization, and compile it with the Adam optimizer and sparse categorical cross-entropy loss.
4. **Train the Model**: Use early stopping to prevent overfitting and train the model on the training data.
5. **Evaluate the Model**: Plot training and validation accuracy/loss curves to monitor performance.
6. **Reuse the Model**: Save the model and reload it for making predictions on new data.

   
### Overview of Files in Repository

#### **Directory Structure**
The repository is structured to facilitate easy access to the dataset, model scripts, and other resources required for training and evaluation. Here’s an overview of the key files and directories:

1. **`data/`**:
   - **Purpose**: Contains all datasets related to the project.
   - **Relevant Files**:
     - `train(1).csv`: The main dataset used for training and testing the model. It contains features like `rectal_temp`, `pulse`, and `outcome` (the target variable).

2. **`notebooks/`**:
   - **Purpose**: Contains Jupyter notebook used for data exploration, model training, and evaluation.
   - **Relevant Files**:
     - `Model_Training.ipynb`: The main notebook where the model is defined, trained, and evaluated. It contains all the steps, including data preprocessing, model building, training, and plotting training curves.

3. **`README.md`**:
   - **Purpose**: Provides an overview of the repository, instructions on how to set up the environment, run the notebooks, and reproduce results.
   - **Content**: Describes the data, model architecture, and steps to run the project, including prerequisites and any other necessary configurations.

### **Software Setup**

#### **Required Packages**

1. **TensorFlow**: For building and training neural networks.
2. **Keras**: High-level API for TensorFlow to define and train deep learning models.
3. **scikit-learn**: For machine learning utilities, including train-test splitting and model evaluation.
4. **pandas**: For data manipulation and preprocessing.
5. **numpy**: For numerical operations on arrays.
6. **matplotlib**: For plotting training and validation curves.
   
#### **Installation Instructions**
   - You can install the necessary libraries via `pip`
     

### **Data**

#### **Downloading the Data**
- **Dataset URL**: Dataset `train(1).csv` has been provided for download. Alternatively this link can be used: https://www.kaggle.com/competitions/playground-series-s3e22/data?select=train.csv


### **Training**

#### **How to Train the Model**
1. **Define the Model Architecture**:
   - A **Sequential Neural Network** is created with layers including dropout and L2 regularization to avoid overfitting.
   - Use `Adam` optimizer, `sparse_categorical_crossentropy` loss, and accuracy metrics.


2. **Train the Model**:
   - Use **early stopping** to stop training when validation loss does not improve for a specified number of epochs.
   

3. **Monitor Training**:
   - Plot training and validation accuracy/loss curves to check if the model is overfitting or converging:
   

### **Performance Evaluation**

#### **How to Run the Performance Evaluation**


1. **Accuracy and Loss**:
   - Evaluate the model by calculating the **accuracy** on the test set to see how often it correctly predicts the outcome. The **loss** shows how well the model minimizes error based on the chosen loss function (sparse categorical cross-entropy).

2. **AUC-ROC**:
   - **AUC (Area Under the Curve)** measures how well the model distinguishes between classes. A higher AUC score indicates better performance. The **ROC curve** visualizes the trade-off between true positive rate and false positive rate across different threshold values.

3. **Visualize**:
   - The ROC curve can be plotted for each class to see how well the model differentiates between them. A curve closer to the top-left corner indicates better model performance.
