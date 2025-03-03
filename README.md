# Breast Cancer Prediction APP

## Description
This application uses a neural network model to predict whether a tumor is malignant or benign based on clinical characteristics. It allows users to input patient data and receive real-time predictions.

## Step 1: Project Setup

a. **I create a directory for the project called `breast_cancer_analysis`.**
   I open my terminal or command prompt and run the following command:
   mkdir breast_cancer_analysis

b. **I change into the newly created directory.**
     cd breast_cancer_analysis

c. **I initialize a Git repository for version control.**
    git init

d. **I create a virtual environment named venv and activate it.**
    python -m venv venv

e. **I activate the virtual environment: (On Windows)**
    venv\Scripts\activate

f. **I install the necessary dependencies**
    pip install pandas numpy scikit-learn streamlit

## Step 2: Data Acquisition and Preparation
a. **I download the dataset**
    I use the breast cancer dataset from sklearn, which is easily accessible.
b. **Data Preparation**
    - I check for null values in the   dataset.
    - I explore descriptive statistics to better understand my data.
    - I normalize the features using MinMaxScaler.

## Step 3: Feature Selection
This step helps me identify the most relevant features that will contribute to my modeling, improving model performance and reducing training time.

    - I separate the features (X) from the target (y).
    - I use SelectKBest to select the best k features based on the chi-squared score.
    - I extract the indices and names of the selected features, along with their scores.

## Step 4: Grid Search CV for Model Optimization
a. **I set up Grid Search CV**
    GridSearchCV is a tool I use to optimize the hyperparameters of a model. In this case, I use MLPClassifier, which is a multi-layer perceptron classifier.
    - I create an instance of MLPClassifier with a maximum number of iterations.
    - I specify the hyperparameters I want to optimize, including hidden layer sizes, activation function, solver, learning rate, regularization parameter, and batch size.
    - I initialize GridSearchCV with the model, parameter grid, and specify the scoring metric (accuracy in this case).
    - I fit the model to the training data.
    - I print the best parameters found and the best cross-validation score.
b. **Best Parameters**
    Activation: relu
    Alpha: 0.0001
    Batch Size: 20
    Hidden Layer Sizes: (50, 50) (two hidden layers with 50 neurons each)
    Learning Rate: constant
    Solver: adam
    Best Cross-Validation Score: 0.9582 (approximately 95.82%)
c. **Interpretation**
    - Effective Model: The cross-validation score indicates that my model performs very well on training data, suggesting it can generalize well to new data.
    - Optimized Parameters: I have found a hyperparameter configuration that maximizes model accuracy.

## Step 5: Implementing an Artificial Neural Network (ANN) Model
    - Load and Normalize: I load the dataset and normalize the features.
    - Feature and Label Definition: I separate the features from the labels.
    - Data Splitting: I split the dataset into training and testing sets.
    - Model Creation: I define the MLPClassifier model with the chosen hyperparameters.
    - Training: I fit the model to the training data.
    - Evaluation: I make predictions and evaluate the model using the confusion matrix and classification report.
b. **Interpretation**
    - High Performance: Precision and recall are very high for both classes, indicating that the model is effective in correctly classifying benign and malignant cases.
    - Confusion Matrix: A low number of false positives and false negatives suggests that the model has a good balance between sensitivity (the ability to identify positives) and specificity (the ability to identify negatives).
c. **Conclusion**
    The neural network model has demonstrated solid performance in classifying tumors in the test dataset, suggesting that it is an effective tool for breast cancer prediction based on the provided clinical characteristics.

## Step 6: Building a Streamlit App Locally
a. **I develop the Streamlit Application**
    Next, I create a basic application that allows users to input data and obtain predictions about breast cancer using the previously trained model.
b. **User Interface**
    - Title and Header: I add a title and header for the application.
    - Input Fields: I create input fields for each feature of the dataset.
    - Prediction Button: When clicked, the button collects user input, normalizes it, and makes a prediction.
    - Display Results: Depending on the prediction, I show a success or error message.
c. **unning the Application**
    To run the app, I save the code in a file named app.py and execute the following command in my terminal: streamlit run app.py#   B r e a s t _ c a n c e r  
 