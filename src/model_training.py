import logging
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import pickle

logdir = 'logs'
os.makedirs(logdir, exist_ok=True)
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file = os.path.join(logdir, 'model_training.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_model(x_train, y_train):
    """Train a logistic regression model."""
    try:
        model = LogisticRegression()
        model.fit(x_train, y_train)
        logger.debug("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
def save_model(model, output_path):
    """Save the trained model to a file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        model_file = os.path.join(output_path, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {model_file}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
def main():
    try:
        # Load preprocessed data
        data = pd.read_csv('features/x_train.csv')
        target = pd.read_csv('features/y_train.csv')
        
        # Train the model
        model = train_model(data, target)
        
        # Save the trained model
        save_model(model, 'models')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()
    