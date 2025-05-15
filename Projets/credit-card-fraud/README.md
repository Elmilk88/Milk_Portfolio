
# 🕵️‍♂️ Credit Card Fraud Detection

This project demonstrates how to detect fraudulent credit card transactions using machine learning. The dataset is highly imbalanced and we apply appropriate techniques to handle this.

## 📂 Files
- `credit_card_fraud_detection.ipynb`: Complete Jupyter notebook with EDA, preprocessing, model training, and evaluation.
- `main.py`: A simple Python script to load the model and make predictions.
- `fraud_model.pkl`: Saved model (to be created by running the notebook).
- `README.md`: Project summary and instructions.

## 🧠 Model
We used a **Random Forest Classifier** with `class_weight='balanced'` to handle class imbalance. The features were scaled using `StandardScaler`.

## 📊 Evaluation
We evaluated the model using:
- **Confusion matrix**
- **Classification report**
- **ROC AUC curve**
- **Feature importance**

## 🏁 How to Run

1. Place the `fraud_model.pkl` and your input CSV file in the same directory.
2. Run the script using Python:

```bash
python main.py
```

Or import and call the `predict_fraud()` function inside any Python program.

## 📝 Dataset
The dataset used is the [Credit Card Fraud Detection Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
