import cv2
import pytesseract
from spellchecker import SpellChecker

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\zhpla\AppData\Local\Programs\Tesseract-OCR"


def process_image(image):
    # Preprocess the image if needed (e.g., resize, denoise, etc.)
    # ...

    # Apply OCR using Tesseract
    text = pytesseract.image_to_string(image)

    return text


def remove_non_text_objects(text):
    # Split the text into individual lines
    lines = text.split("\n")

    # Remove non-text objects
    cleaned_lines = []
    for line in lines:
        # Remove lines with no alphanumeric characters (non-text)
        if any(char.isalnum() for char in line):
            cleaned_lines.append(line)

    # Join the cleaned lines back into a single string
    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text


def spell_check(text):
    spell = SpellChecker()

    # Split the text into individual words
    words = text.split()

    # Correct misspelled words
    corrected_text = []
    for word in words:
        corrected_word = spell.correction(word)
        corrected_text.append(corrected_word)

    # Join the corrected words back into a single string
    corrected_text = " ".join(corrected_text)

    return corrected_text
