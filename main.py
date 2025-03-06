import pytesseract
from PIL import Image

# Set Tesseract path (if not done earlier)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load an image containing text
image = Image.open(r"C:\Users\Sayantan\Pictures\Screenshots\Screenshot 2025-03-04 225916.png")

# Extract text
text = pytesseract.image_to_string(image)

print("Extracted Text:\n", text)
