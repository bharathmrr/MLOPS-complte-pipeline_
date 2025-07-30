import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
import logging

logdir='logs'
os.makedirs(logdir, exist_ok=True)
logger=logging.getLogger("data_ingestion")

logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(logdir, 'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    """load data from a csv file    """
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in the file: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing the file: {file_path}")
        raise
def preprocess_data(df):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    try:
        if df.isnull().values.any():
            logger.warning("Missing values found in the dataset. Filling with mean.")
            df.fillna(df.mean(), inplace=True)
        
        else:
            logger.debug("No missing values found in the dataset.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
def save_data(train_data,test_data, output_path):
    """Save the preprocessed data to a csv file."""
    try:
        raw_data_path= os.path.join(output_path, 'raw_data.csv')
        os.makedirs(output_path, exist_ok=True)
        train_data.to_csv(os.path.join(output_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(output_path, 'test_data.csv'), index=False)
        logger.debug(f"Train and test data saved successfully to {output_path}")

        logger.debug(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise
def main():
    """Main function to execute the data ingestion and preprocessing pipeline."""
    try:
        file_path ='experments/gender_classification_v7.csv'
        output_path = 'output'
        df = load_data(file_path)
        preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, output_path)   
    except Exception as e:
        logger.error(f"An error occurred in the data ingestion pipeline: {e}")
        raise
if __name__ == "__main__":
    main()
