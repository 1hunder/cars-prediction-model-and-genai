import os
import joblib
import numpy as np
import pandas as pd
import base64
from shap_utils import generate_shap_plot_image
from chatgpt_api import generate_explanation_vision

# Loading the model and feature names
MODEL_DIR = "models"
gbr = joblib.load(os.path.join(MODEL_DIR, "gbr_model.pkl"))
gbr_feature_names = joblib.load(os.path.join(MODEL_DIR, "gbr_features.pkl"))

def make_prediction(input_data: dict) -> dict:
    # Transform input data
    input_df = pd.DataFrame([input_data])

    # Process categorical features as done during training
    categorical_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
    X_new = pd.get_dummies(input_df, columns=categorical_cols)
    X_new = X_new.reindex(columns=gbr_feature_names, fill_value=0)
    X_new = X_new.astype("float64")  # this is important!

    # Prediction
    final_pred = gbr.predict(X_new)[0]

    # SHAP plot
    image_path = generate_shap_plot_image(gbr, X_new)

    # Encode the plot in base64
    with open(image_path, "rb") as f:
        shap_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Generate explanation
    explanation_text = generate_explanation_vision(input_data, final_pred, shap_base64)

    return {
        "prediction": final_pred,
        "shap_plot": shap_base64,
        "explanation": explanation_text
    }
