import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to your image
image_path = "C:/Users/Sayantan/Pictures/Screenshots/Screenshot 2025-03-05 103047.png"  # Change this to the actual path of your image

# Perform OCR
result = reader.readtext(image_path, detail=0)

# Print the extracted text
print("Extracted Text:", result)
