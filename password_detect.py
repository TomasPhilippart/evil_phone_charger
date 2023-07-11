#!/usr/bin/python3

# DISCLAIMER
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import argparse
from pathlib import Path
import cv2
import os
import re
import pytesseract
import numpy as np
import easyocr

# Threshold for fine tuning -> feel free to adjust these
BLUR_THRESHOLD = 1000
CONTRAST_THRESHOLD = 200
CONFIDENCE_THRESHOLD_TESSERACT = 70
CONFIDENCE_THRESHOLD_EASYOCR = 0.65
VERTICAL_TOLERANCE = 100
TRADEOFF_RATIO = 0.5

# Characters that can belong in a password
CHARACTER_WHITELIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_+=[]{};:,.<>/?"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir', required=True, type=str, help='Directory where frames are stored.')
parser.add_argument('--interactive', action='store_true', help='Run the program in interactive mode with user input.')
parser.add_argument('--ocr', choices=['tesseract', 'easyocr','google_vision'], default='easyocr', help='Choose the OCR engine to use.')
args = parser.parse_args()

# easy ocr reader
if args.ocr == "easyocr":
    reader = easyocr.Reader(['en'])

def preprocess_images(image_directory):
    # Get the image files and sort them in a natural order
    image_files = sorted(os.listdir(image_directory), key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', x)])
    images = []

    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        images.append(image)

    return images

def calculate_absolute_difference(prev_image, curr_image):
    return cv2.absdiff(prev_image, curr_image)

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def threshold_image(image):
    _, threshold = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    return threshold

def find_largest_contour(image, prev_contour=None):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if prev_contour is None or len(contours) == 0:
        return max(contours, key=cv2.contourArea) if len(contours) > 0 else None

    prev_y = prev_contour[:, 0, 1].mean()  # Mean y-coordinate of previous contour

    # Filter contours within a certain y-coordinate range
    candidate_contours = [contour for contour in contours if abs(contour[:, 0, 1].mean() - prev_y) <= VERTICAL_TOLERANCE]

    if len(candidate_contours) == 0:
        return None

    largest_contour = max(candidate_contours, key=cv2.contourArea)
    largest_contour_area = cv2.contourArea(largest_contour)

    if prev_contour is not None:
        closest_contour = min(candidate_contours, key=lambda contour: abs(contour[:, 0, 1].mean() - prev_y))
        closest_contour_area = cv2.contourArea(closest_contour)

        if 1 - TRADEOFF_RATIO * largest_contour_area > TRADEOFF_RATIO * closest_contour_area:
            return largest_contour

    return closest_contour

def calculate_laplacian_variance(image):
    gray = convert_to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)

def create_contour_mask(image, contour):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
    return mask

def apply_dilation(image, iterations=1):
    kernel = np.ones((15, 15), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

def apply_mask(image, mask):
    return cv2.bitwise_and(image, mask)

def crop_image(image, x, y, w, h):
    return image[y:y + h, x:x + w]

def join_images_horizontally(image_directory, output_path):
    images = []
    max_height = 0

    # Get the image files and sort them based on creation time
    image_files = sorted(os.listdir(image_directory), key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))

    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        if image is not None:
            images.append(image)
            max_height = max(max_height, image.shape[0])

    if len(images) > 0:
        resized_images = []
        for image in images:
            height, width = image.shape[:2]
            scale = max_height / height
            resized_image = cv2.resize(image, (int(width * scale), max_height))
            resized_images.append(resized_image)

        concatenated_image = cv2.hconcat(resized_images)
        cv2.imwrite(output_path, concatenated_image)

    # Delete all files inside the image_directory
    [os.remove(os.path.join(image_directory, file)) for file in os.listdir(image_directory) if
     os.path.isfile(os.path.join(image_directory, file))]

    
def perform_ocr(image):
    if args.ocr == "tesseract":
        text_data = pytesseract.image_to_data(image, config="--psm 10 -l eng --dpi 72", output_type=pytesseract.Output.DICT)
        confidences = text_data["conf"]
        texts = text_data["text"]

        # Filter texts by confidence threshold
        valid_texts = [text for confidence, text in zip(confidences, texts) if confidence > CONFIDENCE_THRESHOLD_TESSERACT]

        if len(valid_texts) > 0:
            max_confidence_text = max(valid_texts, key=lambda x: confidences[texts.index(x)])
            return max_confidence_text
        
    elif args.ocr == "easyocr":
        results = reader.readtext(image)
        valid_texts = [result[1] for result in results if result[2] > CONFIDENCE_THRESHOLD_EASYOCR]

        if len(valid_texts) > 0:
            max_confidence_text = max(valid_texts, key=len)
            return max_confidence_text.strip()
    
    elif args.ocr == "google_vision":
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return texts[0].description
        
    return ""


def save_image(image, path):
    cv2.imwrite(path, image)

def display_image(image, window_name, frame_index, total_frames):
    if args.interactive:
        title = f"Motion-based detection - [{frame_index}/{total_frames}]"
        cv2.moveWindow(window_name, 5,15) # move image to top left
        cv2.imshow(window_name, image)
        cv2.setWindowTitle(window_name, title)
        cv2.waitKey(0)

def main():
    # Directory containing the images
    image_directory = args.frame_dir

    # Preprocess the images
    images = preprocess_images(image_directory)

    prev_image = images[0]
    prev_contour = None
    index = 0

    ocr_text = ""  # Initialize OCR text string

    while index < len(images) - 1:
        
        curr_image = images[index]

        diff = calculate_absolute_difference(prev_image, curr_image)
        gray_diff = convert_to_grayscale(diff)
        threshold = threshold_image(gray_diff)

        contour = find_largest_contour(threshold, prev_contour)

        if contour is not None:
            mask = create_contour_mask(gray_diff, contour)
            dilated_mask = apply_dilation(mask)

            masked_diff = apply_mask(gray_diff, dilated_mask)

            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cropped_diff = crop_image(masked_diff, x, y, w, h)
            else:
                cropped_diff = masked_diff

            cropped_diff = cropped_diff[:, int(cropped_diff.shape[1] * 0.23):-int(cropped_diff.shape[1] * 0.24)]
            cropped_diff_rgb = cv2.cvtColor(cropped_diff, cv2.COLOR_GRAY2BGR)

            comparison_image = cv2.hconcat([prev_image, curr_image])

            mask = cropped_diff == 0
            cropped_diff_rgb[mask] = (0, 0, 0)

            height = max(comparison_image.shape[0], cropped_diff_rgb.shape[0])
            comparison_image = cv2.resize(comparison_image, (0, 0), fx=height/comparison_image.shape[0], fy=height/comparison_image.shape[0])
            cropped_diff_rgb = cv2.resize(cropped_diff_rgb, (0, 0), fx=height/cropped_diff_rgb.shape[0], fy=height/cropped_diff_rgb.shape[0])

            combined_image = np.concatenate((comparison_image, cropped_diff_rgb), axis=1)

            # Blurriness and text detection
            laplacian_var = calculate_laplacian_variance(cropped_diff)
            is_blurry = laplacian_var < BLUR_THRESHOLD
            # Calculate contrast
            contrast = np.max(cropped_diff) - np.min(cropped_diff)

            display_image(combined_image, "Image Comparison", index+1, len(images))

            if not is_blurry:
                text = perform_ocr(cropped_diff).strip()
                prev_contour = contour
                
                if (contrast > CONTRAST_THRESHOLD) and (text in CHARACTER_WHITELIST) and (text != ""):
                    ocr_text += text  # Append OCR text to the string
                    print(f"OCR Text: '{ocr_text}'")

                    # Save corresponding image
                    Path("results/temp").mkdir(parents=True, exist_ok=True)
                    cropped_image_path = f"results/temp/cropped_image_{text}-{index}.jpg"
                    save_image(cropped_diff, cropped_image_path)
        else:
            comparison_image = cv2.hconcat([prev_image, curr_image])
            display_image(comparison_image, "Image Comparison", index+1, len(images))

        if args.interactive:
            key = cv2.waitKey(0)

            if key == ord("q"):
                break
            elif key == ord("d"):
                index = min(index + 1, len(images) - 1)
                prev_image = curr_image
            elif key == ord("a"):
                index = max(index - 1, 0)
                prev_image = curr_image
        else:
            index = min(index + 1, len(images) - 1)
            prev_image = curr_image
            
        cv2.destroyAllWindows()
        
    return ocr_text

if __name__ == "__main__":
    ocr_text = main()
    join_images_horizontally("results/temp", f"results/result_{ocr_text}.jpg")