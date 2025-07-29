import logging
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
logdir = 'logs'
os.makedirs(logdir, exist_ok=True)
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file = os.path.join(logdir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
def load_model(model_path):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
def evaluate_model(model, x_test, y_test):
    """Evaluate the model using various metrics."""
    try:
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        logger.debug(f"Model evaluation results: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
def save_evaluation_results(results, output_path):
    """Save the evaluation results to a file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        results_file = os.path.join(output_path, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logger.debug(f"Evaluation results saved successfully to {results_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise
def main():
    try:
        # Load the model
        model = load_model('models/model.pkl')
        
        # Load test data
        x_test = pd.read_csv('features/x_test.csv')
        y_test = pd.read_csv('features/y_test.csv').values.ravel()  # Flatten the array
        
        # Evaluate the model
        results = evaluate_model(model, x_test, y_test)
        
        # Save evaluation results
        save_evaluation_results(results, 'evaluation_results')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()