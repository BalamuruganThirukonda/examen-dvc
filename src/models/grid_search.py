import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os


def grid_search(X_train, y_train, model_type="RandomForest"):
    """
    Perform GridSearchCV to find the best hyperparameters for the regression model.
    
    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model_type (str): Type of regression model to use ("RandomForest" or "LinearRegression").
    
    Returns:
        best_params (dict): Best hyperparameters found by GridSearchCV.
    
    """
    
    if model_type == "RandomForest":
        model = RandomForestRegressor()  # Use RandomForestRegressor here
        param_grid = {
            "n_estimators" : [100, 200],
            "max_depth" : [10, 20, None],
            "min_samples_split": [2, 5]
        }
    elif model_type == "LinearRegression":
        model = LinearRegression()
        param_grid = {}
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    #Perform GridSearchCV
    logging.info(f"Performing GridSearchCV for {model_type}...")
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    logging.info(f"Best parameters found {best_params}")
    
    return best_params

def save_best_params(best_params, filename="models/best_params.pkl"):
    """
    Save the best hyperparameters to a .pkl file.
    
    Parameters:
        best_params (dict): Best hyperparameters found by GridSearchCV.
        filename (str): Path to save the best parameters.
    """
    logging.info(f"Saving best parameters to {filename}...")
    joblib.dump(best_params, filename)
    logging.info("Best parameters saved successfully.")
    
def main(input_dir = "data/normalized_data", output_dir="models"):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Load training data
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()
    
    # Perform GridSearchCV and get the best parameters
    best_parameters = grid_search(X_train, y_train, model_type="RandomForest")
    
    # Save the best parameters
    save_best_params(best_parameters, os.path.join(output_dir, "best_params.pkl"))
    
if __name__ == "__main__":
    main()