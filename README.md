**# Car Price Prediction with Explainable and Accurate AI

## Author: Viktor Kulyk

---

## Overview
This project is a complete machine learning pipeline for predicting used car prices. It combines classical models, deep learning, and explainability techniques. My goal was to build not only a high-performing predictor but also a system that can explain its decisions using GenAI (e.g., ChatGPT API).

I used real data scraped from [Otomoto.pl](https://www.otomoto.pl/) and built several models, including:
- Interpretable models like Gradient Boosting Regressor (GBR) with SHAP
- High-accuracy models like a stacked ensemble (GBR + 1D CNN)
- Deep learning architectures (FNN, 1D CNN)

The system supports practical deployment in a FastAPI app with future plans for a web-based UI and natural language explanations.

---

## 📦 Project Structure
```
├── data/                 # Raw and cleaned datasets
├── notebooks/            # EDA, training, SHAP notebooks
├── models/               # Saved models (GBR, CNN, stacked)
├── shap/                 # SHAP visualizations
├── api/                  # FastAPI app (WIP)
├── app/                  # Future front-end
├── README.md             # Project overview
```

---

## 📊 Data Collection
I scraped car listings from Otomoto.pl on 03/05/2025. Initial scraping was done using:
- `requests`, `httpx`, and `bs4` for fast HTML parsing
- Later switched to `selenium` in headless & stealth mode to bypass anti-bot protection

Collected ~3,700 listings with features such as:
- Brand, Model, Year
- Mileage, Fuel Type, Transmission
- Engine Size, Horsepower
- Color, Description, Price

---

## 🔧 Data Preprocessing & Feature Engineering
- Cleaned numerical columns (`Mileage`, `Engine Size`, `Price`, etc.)
- Standardized inconsistent formats
- Translated Polish categorical values into English
- Grouped colors into `Light`, `Dark`, `Other`
- Converted `Transmission` to boolean: `IsAutomatic`
- Created `LogPrice` for log regression (used selectively)

---

## 📈 Exploratory Data Analysis (EDA)
- Examined feature distributions
- Detected skewness and outliers (via histograms and boxplots)
- Applied statistical tests (Shapiro-Wilk) to assess normality
- Investigated correlations with price
- Visualized price trends by brand, fuel type, color, year

---

## 🧠 Machine Learning Models
I trained the following models and tuned their parameters:

### 🔹 Gradient Boosting Regressor (GBR)
- Best interpretable model
- Compatible with SHAP for feature attribution

### 🔹 CatBoost, XGBoost, Random Forest
- Tree-based models with good baseline performance

### 🔹 Model Evaluation
| Model                | R²     | MAE     | RMSE    |
|---------------------|---------|---------|---------|
| Random Forest        | 0.8913  | 15,291  | 23,480  |
| CatBoost             | 0.8923  | 15,240  | 23,374  |
| XGBoost              | 0.9163  | 13,372  | 20,071  |
| Gradient Boosting    | 0.9188  | 13,041  | 19,775  |

---

## 🔬 Deep Learning Models

### 🔹 Feedforward Neural Network (FNN)
- Architecture: 128-64-1 with ReLU + dropout
- Trained with RMSLE loss
- R²: 0.81, MAE: ~16,000, RMSE: ~29,000

### 🔹 1D Convolutional Neural Network (CNN)
- Conv1D layers capture local feature interactions
- R²: 0.912, MAE: ~13,600, RMSE: ~20,600

---

## 🤖 Stacked Model (GBR + CNN)
- Combined predictions from GBR and 1D CNN
- Trained meta-model (Linear Regression) on outputs
- Final metrics:
  - R²: ~0.925
  - MAE: ~13,000
  - RMSE: ~20,000

This model achieved the best tradeoff between accuracy and complexity.

---

## 🔍 Explainability with SHAP
I used SHAP to explain GBR predictions:
- Performed global and local explanations
- Visualized why certain cars are priced higher/lower
- Example: SHAP showed `Year`, `Engine Size`, and `Brand` were key drivers

This supports transparent decision-making and can be extended into GenAI interfaces.

---

## 🧪 Real Car Test Cases
Tested predictions for:
1. **Ford Focus (2015)**
   - Real Price: 42,900 PLN
   - GBR: 37,462 PLN
   - Stacked: 40,103 PLN

2. **BMW Seria 5 (2020)**
   - Real Price: 127,000 PLN
   - GBR: 124,011 PLN
   - Stacked: 136,457 PLN

SHAP explanations clearly showed feature contributions to the price.

---

## 🌐 What’s Next: Web App + GenAI Explainability
My next step is to develop a lightweight **web application** that will:
- Allow users to enter car details
- Get predictions from both models
- Use **ChatGPT API** + SHAP to generate explanations like:
  > "Your car is valued at 39,500 PLN because it’s a manual 2014 Ford with above-average mileage."

The final goal is a transparent AI tool that regular users can trust.

---

## 💻 Tech Stack
- Python, Pandas, Scikit-Learn, Keras, PyTorch, XGBoost, CatBoost
- SHAP for explainability
- Matplotlib / Seaborn for visualization
- FastAPI (planned)

---



## Contact
Have feedback or ideas? Reach out on GitHub or email me on [viktor.kulyk@hotmail.com](mailto:viktor.kulyk@hotmail.com)!

**
