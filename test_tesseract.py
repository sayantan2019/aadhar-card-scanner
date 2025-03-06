import pytesseract

# Manually set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    print("Tesseract Version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("Error:", e)
