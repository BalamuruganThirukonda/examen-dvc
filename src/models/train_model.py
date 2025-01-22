import pandas as pd
import joblib
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train, model_type="RandomForest", best_params=None):
    """
    Train the model using the best hyperparameters from GridSearchCV.
    
    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of regression model to use ("RandomForest" or "LinearRegression").
        best_params (dict): Best hyperparameters found from GridSearchCV.
    
    Returns:
        model (sklearn model): Trained model.
    """
    
    if model_type == "RandomForest":
        model = RandomForestRegressor(**best_params)
    elif model_type == "LinearRegression":
        model = LinearRegression(**best_params)
    else:
        raise ValueError(f"Unsupported Model type: {model_type}")
    
    logging.info(f"Training {model_type} model...")
    model.fit(X_train, y_train) 
    
    return model

def save_trained_model(model, filename="model/trained_model.pkl"):
    """
    Save the trained model to a .pkl file.
    
    Parameters:
        model (sklearn model): Trained model to save.
        filename (str): Path to save the model.
    """
    logging.info(f"Saving the trained model to {filename}...")
    joblib.dump(model, filename)
    logging.info("Model saved successfully")
    
def main(input_dir="data/normalized_data", output_dir="models"):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    #Load the training data
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()
    
    #Load the best parameters from GridSearchCV
    best_params = joblib.load(os.path.join(output_dir, "best_params.pkl"))
    
    #Train the model using best parameters
    model = train_model(X_train, y_train, model_type="RandomForest", best_params=best_params)
    
    #Save the trained model
    save_trained_model(model, os.path.join(output_dir, "trained_model.pkl"))

if __name__ == "__main__":
    main()