import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

def split_data(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Load the dataset, split into train and test sets, and save the results.
    
    Parameters:
        input_file (str): Path to the raw data (csv file).
        output_dir (str): Directory to save the processed datasets.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load the dataset
    logging.info(f"Loading dataset from {input_file}")
    data = pd.read_csv(input_file)
    
    # Separate features and target
    y = data["silica_concentrate"]
    X = data.drop(columns=["silica_concentrate", "date"], axis=1)

    
    #Split inot train and test sets
    logging.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    #Save the split datasets
    logging.info("Saving split datasets")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    logging.info(f"Data successfully split and saved to {output_dir}")
    
def main(input_file= "data/raw_data/raw.csv", output_dir = "data/processed_data"):
    """
    Main function to handle data splitting.
    
    Parameters:
        input_file (str) : Path to the input raw data file
        output_dir (str): Path to save the preprocessed file.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    split_data(input_file, output_dir)
    
if __name__ == "__main__":
    main()
    
    