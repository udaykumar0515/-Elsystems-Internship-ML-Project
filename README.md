# 🎓 Elsystems Internship - Machine Learning Developer

**Company:** Elsystems  
**Role:** Machine Learning Intern  
**Project:** Customer Churn Prediction System

---

## 📋 Internship Overview

This repository documents my work as a **Machine Learning Intern** at **Elsystems**. During this internship, I focused on developing predictive models and interactive web applications to solve real-world business problems, specifically targeting customer retention.

The primary objective was to build a robust system capable of analyzing customer behavior and predicting the likelihood of churn, enabling the company to take proactive retention measures.

---

## 🚀 Project: Customer Churn Prediction

### 📖 Description

Customer Churn is a critical metric for any subscription-based business. This project leverages **Machine Learning** to predict whether a customer is likely to cancel their service based on historical data such as tenure, monthly charges, and contract details.

I developed an end-to-end solution including data preprocessing, model training using a **Random Forest Classifier**, and a user-friendly frontend interface built with **Streamlit**.

### 🛠️ Technology Stack

- **Language:** Python 3.x
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Graphviz
- **Web Framework:** Streamlit
- **Model Serialization:** Pickle

---

## 📂 Project Structure

- **`Application.py`**: The main Streamlit web application script. Allows users to input data and get real-time churn predictions.
- **`Model.py`**: The training script. Loads the dataset, preprocesses features (One-Hot Encoding, Label Encoding), trains the Random Forest model, and saves it as `churn_model.pkl`.
- **`EIsystem.ipynb`**: Jupyter Notebook containing exploratory data analysis (EDA), data flow diagrams (DFD), and visualizations of model performance.
- **`Customer-Churn.csv`**: The dataset used for training and testing the model.
- **`docs/`**: Contains internship reports and presentation slides.

---

## 📊 Key Features

1.  **Data Preprocessing**: Handling categorical variables via `get_dummies` and Label Encoding.
2.  **Model Training**: Utilizes a Random Forest Classifier for high accuracy and robustness.
3.  **Interactive Dashboard**: A Streamlit app that takes user inputs (Tenure, Monthly Charges, Contract Type, Payment Method) and displays the churn probability.
4.  **Persistence**: The trained model is saved and loaded dynamically for efficient inference.

---

## 💻 How to Run

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/udaykumar0515/-Elsystems-Internship-ML-Project.git
    cd -Elsystems-Internship-ML-Project
    ```

2.  **Install Dependencies**
    Ensure you have Python installed, then set up the environment:

    ```bash
    pip install pandas scikit-learn streamlit matplotlib seaborn graphviz
    ```

3.  **Train the Model** (Optional, if `churn_model.pkl` is missing)

    ```bash
    python Model.py
    ```

4.  **Run the Application**
    ```bash
    streamlit run Application.py
    ```

---

## 📈 Analysis & Insights

_Detailed analysis and data flow diagrams can be found in `EIsystem.ipynb`._

The model analyzes key factors driving customer churn, identifying that **Contract Type** and **Monthly Charges** are significant predictors. The project demonstrates the practical application of AI in minimizing revenue loss.

---

_This project was developed as part of the internship curriculum at Elsystems._
