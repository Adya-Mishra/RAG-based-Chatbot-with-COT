# utils.py

import logging

def clean_text(text):
    # Function to preprocess and clean text data
    return text.lower().strip()

def log_error(message):
    logging.error(message)
