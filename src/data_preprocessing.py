import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
logger= logging.getLogger('data_preprocessing')
logerdir='logs'
os.makedirs(logerdir, exist_ok=True)
logger.setLevel('DEBUG')

consoule_handler = logging.StreamHandler()
consoule_handler.setLevel('DEBUG')
log_file=os.path.join(logerdir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoule_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(consoule_handler)
logger.addHandler(file_handler)

def preprocess_data_(file_path):
    """
    Preprocess the data from the given file path.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        logger.debug(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.debug("Data loaded successfully")
        data['gender'] = data['gender'].map({'Male': 0,'Female': 1})
        logger.debug("Converted categorical successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_data(data, target):
    """
    Split the data into training and testing sets.
    """
    from sklearn.model_selection import train_test_split
    X=data.drop(columns=[target])
    y=data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42 )
    logger.debug("Data split into training and testing sets")
    return x_train, x_test, y_train, y_test
def preprocess_data(precessfile,data):
    """
    Preprocess the data from the given file path.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        os.makedirs(precessfile, exist_ok=True)
        data.to_csv(os.path.join(precessfile, 'preprocessed_data.csv'), index=False)
        logger.debug(f"Preprocessed data saved to {os.path.join(precessfile, 'preprocessed_data.csv')}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        raise
def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    try:
        file_path = 'output/train_data.csv'
        precessfile = 'preprocessed_data'
        data = preprocess_data_(file_path)
        x_train, x_test, y_train, y_test = split_data(data,'gender')

        data_=data
        preprocess_data(precessfile, data_)
        logger.debug("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        raise
if __name__ == "__main__":
    main()
import logging
import pandas as pd
