import pandas as pd
import joblib
import logging
import os
from sklearn.metrics import mean_squared_error, r2_score
import json

def evaluate_model(X_test, y_test, model):
    """
    Evaluate the model using test data and calculate performance metrics.
    
    Parameters:
        X_test (DataFrame): Test features.
        y_test (Series): Test target.
        model (sklearn model): Trained model.
    
    Returns:
        metrics (dict): Model evaluation metrics (MSE, R2).
        predictions (DataFrame): Model predictions on test set.
    """
    logging.info("Making predictions on the test set...")
    predictions = model.predict(X_test)
    
    #Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "mean_squared_error": mse,
        "r2_score": r2
    }
    
    return metrics, predictions

def save_metrics(metrics, filename="metrics/scores.json"):
    """
    Save model evaluation metrics to a JSON file.
    
    Parameters:
        metrics (dict): Model evaluation metrics.
        filename (str): Path to save the metrics.
    """
    logging.info(f"Saving metrics to {filename}...")
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved successfully")
    
def save_predictions(predictions, filename="metrics/predictions.csv"):
    """
    Save model predictions to a CSV file.
    
    Parameters:
        predictions (DataFrame): Model predictions on the test set.
        filename (str): Path to save the predictions.
    """
    logging.info(f"Saving predictions to {filename}...")
    pd.DataFrame(predictions, columns=["Predicted_Silica_concentrate"]).to_csv(filename, index=False)
    logging.info("Predictions saved successfully.")
    
def main(input_dir="data/normalized_data", output_dir="metrics", model_dir="models"):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load the test data
    X_test = pd.read_csv(os.path.join(input_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).values.ravel()
    
    #Load the trained model
    model = joblib.load(os.path.join(model_dir, "trained_model.pkl"))
    
    #Evaluate the model and get metrics and predictions
    metrics, predictions = evaluate_model(X_test, y_test, model)
    
    # Create output directory if doesn't exist    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Save metrics and predictions
    save_metrics(metrics, os.path.join(output_dir, "scores.json"))
    save_predictions(predictions, os.path.join(output_dir, "predictions.csv"))
    

if __name__ == "__main__":
    main()