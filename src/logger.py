import logging
import os
from datetime import datetime

# Create a log file with the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Set the log file path to the CRM directory with a "Logs" folder
logs_path = os.path.join("C:\\Users\\Harshali\\Documents\\CRM", "Logs")

# Create the "Logs" folder if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Set the log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)