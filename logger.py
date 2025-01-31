import logging
import os
import config

# Ensure log directory exists
os.makedirs(config.LOG_PATH, exist_ok=True)

# Define log format
log_format = "[%(asctime)s] [%(levelname)s] - %(message)s"

# Configure logging to support UTF-8
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(f"{config.LOG_PATH}/app.log", encoding="utf-8"),  # Log to file with UTF-8
        logging.StreamHandler()  # Also print logs to console
    ]
)

logger = logging.getLogger(__name__)