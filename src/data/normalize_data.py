import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging

def normalize_data(input_dir, output_dir):
    """
    Normalize the training and testing datasets using StandardScaler.
    
    Parameters:
        input_dir (str): Directory containing X_train.csv and X_train.csv#
        output_dire (str): Directory to save the normalized datasets
    """
    
    #Create output directory if it doesnt exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Load the training and testing data
    logging.info("Loading training and testing datasets")
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
    
    
    # Identify numeric columns for scaling
    numeric_columns = X_train.select_dtypes(include=["number"]).columns
    non_numeric_columns = X_train.columns.difference(numeric_columns)
    
    #Initialize the scaler
    scaler = StandardScaler()
    
    #Fit the scaler on the training data and transform both train and test datasets
    logging.info("Normalizing the datasets")
    X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled_numeric = scaler.transform(X_test[numeric_columns])
    
    #Convert the scaled numeric arrays back to Dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_numeric, columns=numeric_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_numeric, columns=numeric_columns)
    
    #Reattach non-numeric columns to the scaled data
    X_train_scaled = pd.concat([X_train_scaled, X_train[non_numeric_columns].reset_index(drop=True)], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test[non_numeric_columns].reset_index(drop=True)], axis=1)
    
    #Save the normalized datasets
    logging.info("Saving normalized datasets..")
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    logging.info(f"Normalized datasets saved to {output_dir}")
    
def main(input_dir="data/processed_data", output_dir="data/normalized_data"):
    """
    Main function to normalize data and save the output.
    
    Parameters: 
        input_dir (str): Path to the directory containing raw processed data.
        output_dir (str): Path to save the normalized data
        
    """
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    normalize_data(input_dir, output_dir)
    
if __name__ == "__main__":
    main()