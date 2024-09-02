import numpy as np 
import pandas as pd 
import os

# Print all filenames in the input directory (example, adjust as needed)
for dirname, _, filenames in os.walk('D:\\Sem 4\\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pytesseract
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load and display the second image
img = mpimg.imread('D:\\Sem 4\\data\\029\\a02-053.png')
plt.figure(figsize=(10,10))
plt.imshow(img, cmap='gray')

# Use OpenCV to read the second image and perform OCR
img = cv2.imread('D:\\Sem 4\\data\\029\\a02-053.png')
text = pytesseract.image_to_string(img)
print(text)
