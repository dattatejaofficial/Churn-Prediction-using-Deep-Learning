# Churn Prediction using Deep Learning

This project focuses on predicting customer churn using deep learning techniques, specifically Artificial Neural Networks (ANN). Customer churn refers to the loss of clients or customers, and accurately predicting it allows businesses to implement strategies to retain valuable customers.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Extracted Insights](#extracted-insights)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The project utilizes a dataset containing various customer attributes to train an ANN model capable of predicting the likelihood of a customer leaving the company. By analyzing patterns within the data, the model identifies factors contributing to customer churn.

**Key Steps in the Analysis:**

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Handle categorical variables through encoding.
   - Split the data into training and testing sets.
   - Scale features to ensure uniformity.

2. **Model Building:**
   - Construct an ANN using TensorFlow and Keras.
   - Define the architecture with input, hidden, and output layers.
   - Compile the model with appropriate loss functions and optimizers.

3. **Model Training:**
   - Train the model on the training data.
   - Monitor performance metrics during training.

4. **Model Evaluation:**
   - Evaluate the model's performance on the test set.
   - Use metrics such as accuracy, precision, recall, and F1-score.

5. **Prediction:**
   - Utilize the trained model to predict customer churn.
   - Interpret the results to identify at-risk customers.

## Project Structure

The repository contains the following files:

- `Churn_Modelling.csv`: The dataset used for training and testing the ANN.
- `artificial_neural_network(Customer_Churn_Prediction).ipynb`: A Jupyter Notebook that includes code for data preprocessing, model building, training, and evaluation.
- `requirements.txt`: A text file listing the necessary Python packages for the project.
- `app.py`: A Python script for deploying the model using Streamlit, providing an interactive web interface for predictions.
- `model.keras`: The trained ANN model saved in Keras format for deployment.
- `scaler.pkl`: A pickle file containing the scaler used for feature scaling.
- `label_encoder_gender.pkl`: A pickle file containing the label encoder for the 'Gender' feature.
- `onehot_encoder_geo.pkl`: A pickle file containing the one-hot encoder for the 'Geography' feature.
- `predictions.ipynb`: A Jupyter Notebook for making predictions using the trained model.

## Setup Instructions

To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   Use the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/dattatejaofficial/Churn-Prediction-using-Deep-Learning.git
   ```

2. **Navigate to the Project Directory:**
   Move into the project directory:

   ```bash
   cd Churn-Prediction-using-Deep-Learning
   ```

3. **Create a Virtual Environment (optional but recommended):**
   Set up a virtual environment to manage project dependencies:

   ```bash
   python3 -m venv env
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the analysis and make predictions:

1. **Ensure the Virtual Environment is Activated:**
   Make sure your virtual environment is active (refer to the setup instructions above).

2. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open `artificial_neural_network(Customer_Churn_Prediction).ipynb` in the Jupyter interface and execute the cells sequentially to perform the analysis.

3. **Run the Streamlit App:**
   For an interactive web interface to make predictions:

   ```bash
   streamlit run app.py
   ```

   This will start a local server, and you can access the app in your web browser.

## Extracted Insights

The analysis provides valuable insights into customer churn:

- **Model Performance:** The ANN model achieves an accuracy of approximately 86.3% on the test set, indicating a reliable prediction capability.

- **Feature Importance:** By analyzing the model, we can identify key factors influencing customer churn, such as account balance, tenure, and product usage.

- **Customer Retention:** Identifying at-risk customers allows the company to implement targeted retention strategies, potentially reducing churn rates.

## Dependencies

The project requires the following Python packages:

- `pandas`
- `numpy`
- `tensorflow`
- `keras`
- `scikit-learn`
- `streamlit`

These dependencies are essential for data manipulation, model building, and deployment.
