import logging
import os

logging_level = getattr(logging, os.environ.get('LOG_LEVEL', '').upper(), logging.INFO)

logging.basicConfig(level=logging_level)
logging.getLogger("httpx").setLevel(logging.WARNING)
