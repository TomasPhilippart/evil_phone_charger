import cv2
import os
import re
import pytesseract
import numpy as np

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

# Directory containing the images
image_directory = "media/instagram_login-frames"

# Get the image files and sort them in a natural order
image_files = sorted(os.listdir(image_directory), key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', x)])

# Read the first image
prev_image = cv2.imread(os.path.join(image_directory, image_files[0]))

index = 0
while True:
    # Read the current image
    curr_image = cv2.imread(os.path.join(image_directory, image_files[index]))

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(prev_image, curr_image)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale difference image
    _, threshold = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask of the largest contour
        mask = np.zeros_like(gray_diff)
        cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

        # Apply dilation to enlarge the largest contour
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply the mask to the grayscale difference image
        masked_diff = cv2.bitwise_and(gray_diff, dilated_mask)

        # Crop the masked grayscale difference image
        cropped_diff = masked_diff[:, 5:-5]

        # Convert the cropped grayscale difference image to a 3-channel image
        cropped_diff_rgb = cv2.cvtColor(cropped_diff, cv2.COLOR_GRAY2BGR)

        # Create a side-by-side comparison image
        comparison_image = cv2.hconcat([prev_image, curr_image])

        # Paint the background of the cropped difference image in light yellow color
        mask = cropped_diff == 0
        cropped_diff_rgb[mask] = (255, 255, 153)  # Light yellow color (BGR)

        # Create a combined image with the comparison and cropped difference
        combined_image = cv2.hconcat([comparison_image, cropped_diff_rgb])

        # Display the combined image with the painted background
        cv2.imshow("Image Comparison", combined_image)
    else:
        # Display the original comparison image if no contours found
        comparison_image = cv2.hconcat([prev_image, curr_image])
        cv2.imshow("Image Comparison", comparison_image)

    key = cv2.waitKey(0)

    if key == ord("q"):
        break
    elif key == ord("d"):
        index = min(index + 1, len(image_files) - 1)
        prev_image = curr_image
    elif key == ord("a"):
        index = max(index - 1, 0)
        prev_image = curr_image
    elif key == ord(" "):  # Press spacebar to perform OCR
        if len(contours) > 0:
            # Perform OCR using pytesseract on the cropped difference image
            text = pytesseract.image_to_string(cropped_diff, config="--psm 10 -l eng")
            print("OCR Text:")
            print(text)
        else:
            print("No contours found.")

cv2.destroyAllWindows()
