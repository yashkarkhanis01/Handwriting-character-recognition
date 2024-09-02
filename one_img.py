import pytesseract
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Ensure Tesseract is in your PATH or specify the path directly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the path to your image file
image_path = 'D:\\Sem 4\\data\\self_lines\\selfMade_126.png'

# Read and display the image using Matplotlib
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.show()

# Read the image using OpenCV
img_cv = cv2.imread(image_path)

img=cv2.imread('D:\\Sem 4\\data\\self_lines\\selfMade_126.png')
text = pytesseract.image_to_string(img)
print(text)
