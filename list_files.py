import numpy as np  # For numerical operations
import pandas as pd  # For data processing and CSV file I/O
import os

# Define the local path to your dataset
dataset_path = 'D:\\Sem 4\\data\\'

# List all files in the dataset directory
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
