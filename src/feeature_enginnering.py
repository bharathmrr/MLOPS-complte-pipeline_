import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split

logdir = 'logs'
os.makedirs(logdir, exist_ok=True)
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_path = os.path.join(logdir, 'feature_engineering.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def feature_engineering(file_path):
    try:
        data= pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        logger.debug("Starting feature engineering")
        return data
        # Example feature engineering: creating a new feature
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise
def split_data(data, target):
    """
    Split the data into training and testing sets.
    """

    X = data.drop(columns=[target])
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.debug("Data split into training and testing sets")
    return x_train, x_test, y_train, y_test
def save_features(x_train, y_train,x_test,y_test, output_path):
    try:
        os.makedirs(output_path, exist_ok=True)
        x_train.to_csv(os.path.join(output_path, 'x_train.csv'), index=False)
        y_train.to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
        x_test.to_csv(os.path.join(output_path, 'x_test.csv'), index=False)
        y_test.to_csv(os.path.join(output_path, 'y_test.csv'), index=False)
        logger.debug(f"Features saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving features: {e}")
        raise
def main():
    try:
        file='preprocessed_data/preprocessed_data.csv'  # Replace with your actual file path
        data= feature_engineering(file)
        x_train, x_test, y_train, y_test = split_data(data, target='gender')
        output_path = 'features'
        save_features(x_train, y_train, x_test, y_test, output_path)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()
