
import pandas as pd
from src.logger import logging
from src.pipeline.PredictPipeline import PredictPipeline

# Define the mapping for 'Approved_Flag'
approved_flag_map = ['P1', 'P2', 'P3', 'P4']

# Load your unseen data
data_path = 'C:\\Users\\Harshali\\Documents\\CRM\\notebooks\\data\\Unseen_Dataset_.xlsx'
output_path = 'artifacts\\Updated_Unseen_Dataset.xlsx'

try:
    data = pd.read_excel(data_path)  # Load your unseen data
    logging.info(f"Unseen data loaded from {data_path}")

    # Initialize the prediction pipeline
    pipeline = PredictPipeline()

    # Make predictions
    predictions = pipeline.predict(data)
    logging.info("Predictions made successfully")

    # Map numerical predictions back to 'P1', 'P2', 'P3', 'P4' using the approved_flag_map
    decoded_predictions = [approved_flag_map[int(pred)] for pred in predictions]
    logging.info("Predictions decoded successfully")

    # Combine predictions with the original data
    data['Predictions'] = decoded_predictions

    # Save the updated data to a new Excel file
    data.to_excel(output_path, index=False)  # Save updated data to Excel
    logging.info(f"Updated data with predictions saved to '{output_path}'")

except Exception as e:
    logging.error(f"An error occurred: {e}")

