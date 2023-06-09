from PIL import Image

import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Simple image to string
print(pytesseract.image_to_string(Image.open('test.jpeg')))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
#print(pytesseract.image_to_string('test.jpeg'))

# List of available languages
#print(pytesseract.get_languages(config=''))


# Timeout/terminate the tesseract job after a period of time
try:
    print(pytesseract.image_to_string('test.jpeg', timeout=2)) # Timeout after 2 seconds
except RuntimeError as timeout_error:
    # Tesseract processing is terminated
    pass

# Get bounding box estimates
#print(pytesseract.image_to_boxes(Image.open('test.jpeg')))

# Get verbose data including boxes, confidences, line and page numbers
#print(pytesseract.image_to_data(Image.open('test.jpeg')))

# Get information about orientation and script detection
#print(pytesseract.image_to_osd(Image.open('test.jpeg')))

# Get a searchable PDF
#pdf = pytesseract.image_to_pdf_or_hocr('test.jpeg', extension='pdf')
#with open('test.pdf', 'w+b') as f:
#    f.write(pdf) # pdf type is bytes by default

# Get HOCR output
#hocr = pytesseract.image_to_pdf_or_hocr('test.jpeg', extension='hocr')

# Get ALTO XML output
#xml = pytesseract.image_to_alto_xml('test.jpeg')l