from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import easyocr
import pytesseract
import cv2
import numpy as np
import re
import os
import mysql.connector
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PHOTO_FOLDER'] = 'photos'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PHOTO_FOLDER'], exist_ok=True)

# Initialize OCR engines
reader = easyocr.Reader(['en', 'hi'])
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update for Windows users

# Database connection
try:
    db = mysql.connector.connect(
        host="sql12.freesqldatabase.com",
        user="sql12765927",
        password="CmUkKq2WNd",
        database="sql12765927",
        port=3306
    )
    cursor = db.cursor()
except mysql.connector.Error as err:
    print(f"Database Error: {err}")
    exit(1)

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS extracted_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(255),
        date_of_birth DATE,
        sex ENUM('Male', 'Female', 'Other'),
        aadhaar_number VARCHAR(12),
        photo_path VARCHAR(255),
        text LONGTEXT
    )
""")

def preprocess_image(image_path):
    """Enhances image quality for better OCR accuracy."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to enhance text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    
    # Dilation to fill gaps
    kernel = np.ones((1, 1), np.uint8)
    processed_img = cv2.dilate(thresh, kernel, iterations=1)
    
    return processed_img

def extract_text(image_path):
    """Performs hybrid OCR using both EasyOCR and Tesseract."""
    processed_image = preprocess_image(image_path)
    
    # OCR with Tesseract
    text_tesseract = pytesseract.image_to_string(processed_image, lang='eng')

    # OCR with EasyOCR
    text_easyocr = " ".join(reader.readtext(image_path, detail=0))

    # Merge results
    full_text = f"{text_easyocr} {text_tesseract}"
    
    return full_text

def extract_details(text):
    """Extracts Name, DOB, Gender, and Aadhaar Number using regex."""
    details = {"Full Name": "", "DOB": "", "Sex": "", "Aadhaar Number": ""}

    # Extract Name
    name_match = re.search(r'(?:(?:Name|नाम)[:\s]*)([A-Z\s]+)', text, re.IGNORECASE)
    if name_match:
        details["Full Name"] = name_match.group(1).strip().title()

    # Extract DOB with improved pattern matching and conversion
    dob_match = re.search(r'(?:DOB|Date of Birth|जन्मतिथि)[:\s]*([०-९\d]{1,2}[/\-\.][०-९\d]{1,2}[/\-\.][०-९\d]{4})', text)
    if dob_match:
        dob = dob_match.group(1)
        # Convert Hindi numerals to English
        hindi_to_english = str.maketrans('०१२३४५६७८९', '0123456789')
        dob = dob.translate(hindi_to_english)
        # Normalize separators to '/'
        dob = re.sub(r'[-\.]', '/', dob)
        # Ensure two digits for day and month
        day, month, year = dob.split('/')
        dob = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        details["DOB"] = dob

    # Extract Gender with Hindi support
    gender_match = re.search(r'\b(Male|Female|Other|पुरुष|महिला|अन्य)\b', text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).lower()
        # Convert Hindi gender terms to English
        gender_map = {
            'पुरुष': 'Male',
            'महिला': 'Female',
            'अन्य': 'Other'
        }
        details["Sex"] = gender_map.get(gender, gender.capitalize())

    # Extract Aadhaar Number
    aadhaar_match = re.search(r'\b\d{4} \d{4} \d{4}\b', text)
    if aadhaar_match:
        details["Aadhaar Number"] = aadhaar_match.group(0).replace(" ", "")

    return details

def extract_face(image_path):
    """Extracts face from Aadhaar card image and saves it."""
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Take the first detected face
            x, y, w, h = faces[0]
            
            # Extract face with some margin
            margin = 20
            face = img[max(0, y-margin):min(y+h+margin, img.shape[0]), 
                      max(0, x-margin):min(x+w+margin, img.shape[1])]
            
            # Generate unique filename
            photo_filename = f"face_{uuid.uuid4()}.jpg"
            photo_path = os.path.join(app.config['PHOTO_FOLDER'], photo_filename)
            
            # Save the face image
            cv2.imwrite(photo_path, face)
            return photo_path
        
        return None
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/photos/<filename>')
def photo_file(filename):
    return send_from_directory(app.config['PHOTO_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    
    if file:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract photo
        photo_path = extract_face(filepath)
        if photo_path:
            photo_url = url_for('photo_file', filename=os.path.basename(photo_path))
        else:
            photo_url = None

        # Perform OCR
        full_text = extract_text(filepath)
        print(f"Extracted text: {full_text}")
        
        # Extract Aadhaar details
        extracted_details = extract_details(full_text)
        print(f"Extracted details: {extracted_details}")
        
        if not full_text.strip() or not any(extracted_details.values()):
            # If extracted text is empty, render a form for manual input
            print("Extracted text is empty, rendering manual input form")
            return render_template('manual_input.html', 
                                image_url=filepath,
                                photo_url=photo_path)
        
        print("Rendering preview template")
        return render_template('preview.html',
                            image_url=url_for('uploaded_file', filename=filename),
                            photo_url=photo_url,
                            extracted_text=full_text,
                            full_name=extracted_details["Full Name"],
                            date_of_birth=extracted_details["DOB"],
                            sex=extracted_details["Sex"],
                            aadhaar_number=extracted_details["Aadhaar Number"])

@app.route('/save', methods=['POST'])
def save_text():
    try:
        # Get form data with default values
        full_name = request.form.get('full_name', '').upper()
        date_of_birth = request.form.get('date_of_birth', '')
        sex = request.form.get('sex', '').capitalize()
        aadhaar_number = request.form.get('aadhaar_number', '')
        text = request.form.get('text', '')

        # Validate required fields
        if not all([full_name, date_of_birth, sex, aadhaar_number]):
            print("Missing required fields")
            return redirect(url_for('index'))

        # Validate date format (DD/MM/YYYY)
        try:
            day, month, year = date_of_birth.split('/')
            formatted_date = f"{year}-{month}-{day}"
        except ValueError:
            print("Invalid date format")
            return redirect(url_for('index'))

        # Validate sex value
        valid_sex_values = ['Male', 'Female', 'Other']
        if sex not in valid_sex_values:
            print(f"Invalid sex value: {sex}")
            return redirect(url_for('index'))

        # Validate Aadhaar number (12 digits)
        if not re.match(r'^\d{12}$', aadhaar_number):
            print("Invalid Aadhaar number format")
            return redirect(url_for('index'))

        # Check for duplicate Aadhaar number
        cursor.execute("SELECT id FROM extracted_data WHERE aadhaar_number = %s", (aadhaar_number,))
        if cursor.fetchone():
            print(f"Aadhaar number {aadhaar_number} already exists in database")
            return render_template('error.html', 
                                message=f"Aadhaar card with number {aadhaar_number} already exists in the database")

        # Database insert with formatted date
        cursor.execute("""
            INSERT INTO extracted_data (full_name, date_of_birth, sex, aadhaar_number, photo_path, text)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (full_name, formatted_date, sex, aadhaar_number, request.form.get('photo_url'), text))
        db.commit()
        print("Data saved successfully")

    except mysql.connector.Error as err:
        print(f"Database Insert Error: {err}")
        return render_template('error.html', message="Database error occurred")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return render_template('error.html', message="An unexpected error occurred")
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
