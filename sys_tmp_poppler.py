import sys
from pdf2image import convert_from_path

pdf_file = r"c:\Users\Uday Chalamala\OneDrive\Desktop\Crime_Monitoring\test\Articles-13_06_2025.pdf"
try:
    images = convert_from_path(pdf_file, first_page=1, last_page=1)
    print("SUCCESS: pdf2image works. Got", len(images), "images.")
except Exception as e:
    print("ERROR:", e)
