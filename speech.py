import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image, ImageTk
import cv2
import pytesseract
import os
from langdetect import detect
from language_tool_python import LanguageTool
from englisttohindi.englisttohindi import EngtoHindi
from gtts import gTTS
import pygame
import uuid

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yash's Recognition System")
        self.root.geometry("1500x800")

        # Initialize pygame mixer
        pygame.mixer.init()

        # Initialize recognized_text and corrected_text attributes
        self.recognized_text = ""
        self.corrected_text = ""

        # Load the background image
        self.background_image = Image.open("background.png")
        self.background_image = self.background_image.resize((1563, 864), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Set the background label
        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(x=0, y=0)

        # Upload button
        self.upload_btn = tk.Button(self.root, text="Browse", font=('Helvetica 13 bold'), command=self.upload_image, width=20, height=1)
        self.upload_btn.place(x=310, y=560)

        # Recognized Text Label
        self.recognized_text_label = tk.Label(self.root, text="Recognized Text", font=("Times New Roman", 14))
        self.recognized_text_label.place(x=348, y=610)

        # Text box to display OCR results
        self.text_box = Text(self.root, font=('Helvetica 20 bold'), width=40, height=2, fg="red")
        self.text_box.place(x=120, y=650)

        # Grammar button
        self.grammar_btn = tk.Button(self.root, text="Grammar", font=('Helvetica 13 bold'), command=self.click_g, width=20, height=1)
        self.grammar_btn.place(x=1050, y=300)

        # Text to Speech (Recognized) Button
        self.tts_recognized_btn = tk.Button(self.root, text="Text to Speech (Recognized)", font=('Helvetica 13 bold'), command=self.click_s, width=25, height=1)
        self.tts_recognized_btn.place(x=900, y=650)

        # Text to Speech (Grammar) Button
        self.tts_grammar_btn = tk.Button(self.root, text="Text to Speech (Grammar)", font=('Helvetica 13 bold'), command=self.click_ss, width=22, height=1)
        self.tts_grammar_btn.place(x=1180, y=650)

        # Translate (Recognized) Button
        self.translate_recognized_btn = tk.Button(self.root, text="Translate (Recognized)", font=('Helvetica 13 bold'), command=self.click_t, width=20, height=1)
        self.translate_recognized_btn.place(x=1050, y=400)

        # Translate (Grammar) Button
        self.translate_grammar_btn = tk.Button(self.root, text="Translate (Grammar)", font=('Helvetica 13 bold'), command=self.click_tg, width=20, height=1)
        self.translate_grammar_btn.place(x=1050, y=500)

        # Loading Label
        self.loading_label = tk.Label(self.root, text="Loading....", font=("Times New Roman bold", 14))
        self.loading_label.place(x=350, y=400)

        # Image label to display uploaded images
        self.image_label = tk.Label(self.root)
        self.image_label.place(x=140, y=300)

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
            self.recognized_text = pytesseract.image_to_string(img_preprocessed, config=custom_config)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, self.recognized_text)

    def click_g(self):
        my_tool = LanguageTool('en-US')
        self.corrected_text = my_tool.correct(self.recognized_text)
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, self.corrected_text)

    def click_t(self):
        message = self.recognized_text
        res = EngtoHindi(message)
        op = res.convert
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, op)

    def click_tg(self):
        message = self.corrected_text
        res = EngtoHindi(message)
        ops = res.convert
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, ops)

    def click_s(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            filename = f"recognized_{uuid.uuid4()}.mp3"
            tts = gTTS(text=self.recognized_text, lang='en', slow=False)
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
        except PermissionError:
            print(f"Permission denied: {filename}")

    def click_ss(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            filename = f"corrected_{uuid.uuid4()}.mp3"
            tts = gTTS(text=self.corrected_text, lang='en', slow=False)
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
        except AttributeError:
            print("'corrected_text' is not set")
        except PermissionError:
            print(f"Permission denied: {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
