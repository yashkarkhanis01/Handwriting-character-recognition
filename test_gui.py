import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image, ImageTk
import cv2
import pytesseract
import os

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yash's Recognition System")
        self.root.geometry("1500x800")

        # Load the background image
        self.background_image = Image.open("background.png")
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Set the background label
        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        # Frame to center the upload button
        self.frame = tk.Frame(root, bg='white', bd=5)
        self.frame.place(relx=0.5, rely=0.2, anchor='n', relwidth=0.3, relheight=0.1)  # Adjusted rely value

        # Upload button
        self.upload_btn = tk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(expand=True)

        # Image label to display uploaded images
        self.image_label = tk.Label(root)
        self.image_label.place(relx=0.5, rely=0.5, anchor='n')  # Adjusted rely value for image_label

        # Text box to display OCR results
        self.text_box = Text(root, height=10, width=50)
        self.text_box.place(relx=0.5, rely=0.75, anchor='n')  # Adjusted rely value for text_box

    def resize_image(self, img, max_width=800, max_height=600):
        h, w, _ = img.shape
        if h > max_height or w > max_width:
            ratio = min(max_width / w, max_height / h)
            new_size = (int(w * ratio), int(h * ratio))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Dilation and erosion to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed_img = cv2.dilate(thresh, kernel, iterations=1)
        processed_img = cv2.erode(processed_img, kernel, iterations=1)
        
        return processed_img

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            img = self.resize_image(img)
            img_preprocessed = self.preprocess_image(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # Use --psm 3 (Default): Fully automatic page segmentation, but no OSD (Orientation and Script Detection)
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(img_preprocessed, config=custom_config)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
