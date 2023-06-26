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
import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
import os
import re
import cv2
import time
from moviepy.editor import VideoFileClip
import hashlib
from tqdm import tqdm
from fuzzywuzzy import fuzz
import shutil
import json


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="video", type=str, choices=['usb', 'video'], help='Either capture frames directly from USB device or use a video file.')
parser.add_argument('--capture_time', default="100", type=int, help='How many seconds to capture. Unlimited is 0 (default: 10).')
parser.add_argument('--filename', required=True, type=str, help='Filename for the video to process.')
parser.add_argument('--ocr_mode', default='tesseract', choices=['tesseract', 'google_vision', 'easyocr'], help='Use either Tesseract (default) or Google Cloud Vision API (paid!)')
parser.add_argument('--framerate', default='20', type=int, choices=range(1, 31), help='Framerate for capture (default: 20).')
parser.add_argument('--keywords', type=str, nargs='+', help='Keywords to look for and trim the video based on (example: password,email,confidential)')
args = parser.parse_args()

print(args.keywords)
# (OPTIONAL) Cleanup: remove garbage, easier for testing
os.system('./cleanup.sh')

# EasyOCR reader
if args.ocr_mode == "easyocr":
    reader = easyocr.Reader(['en'])
    print("Warning: Using EasyOCR. Although it yields better results, it is quite slow.")
    
def frame_capture():
    args.filename = f"media/{args.filename}"  # Change the filename to include the media/ prefix, so we store frames there
    print("Starting frame capture... Press CTRL+C to stop recording frames.")

    # Create the directory
    os.makedirs(f"{args.filename}-frames", exist_ok=True)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Unable to read the camera feed")

    # Default resolutions of the frame are obtained. The default resolutions are system-dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    i = 0
    time_start = time.time()
    try:
        while ((args.capture_time != 0 and time.time() < time_start + args.capture_time) or (args.capture_time == 0)):
            ret, frame = cap.read()

            if ret:
                cv2.imwrite(f"{args.filename}-frames/frame{i}.jpg", frame)
                i += 1

            # Break the loop
            else:
                break

            time.sleep(1 / args.framerate)
    except KeyboardInterrupt:
        print("CTRL+C detected, stopping frame capture.")
        pass
    # When everything is done, release the video capture and video write objects
    cap.release()

    # Close all the frames
    cv2.destroyAllWindows()

    return i


def video2frames():
    # Load the video clip
    video_clip = VideoFileClip(args.filename)
    # Make a folder by the name of the video file
    filename, _ = os.path.splitext(args.filename)
    filename += "-frames"
    os.makedirs(filename, exist_ok=True)

    i = 0

    # If the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, args.framerate)
    # If SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # Iterate over each possible frame
    for current_duration in tqdm(np.arange(0, video_clip.duration, step), desc="Converting video to frames", unit="frame"):
        # Format the file name and save it
        frame_filename = os.path.join(filename, f"frame{i}.jpg")
        i += 1
        # Save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)

    return i


def detect_text(path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return texts[0].description


def filter_duplicate_frames(frame_directory):
    file_dict = {}
    duplicate_frames = 0

    # Count the total number of files in the directory
    total_files = sum(1 for _ in Path(frame_directory).glob('*'))

    with tqdm(total=total_files, unit='frame', desc="Filtering duplicate frames") as pbar:
        for video_frame in Path(frame_directory).glob('*'):
            if video_frame.is_file():
                frame_path = str(video_frame)

                # Calculate the SHA checksum of the frame
                with open(frame_path, 'rb') as f:
                    frame_hash = hashlib.sha256(f.read()).hexdigest()

                # Check if the checksum already exists in the dictionary
                if frame_hash in file_dict:
                    # Duplicate frame found
                    os.remove(frame_path)
                    duplicate_frames += 1
                else:
                    # Add the frame to the dictionary with its checksum
                    file_dict[frame_hash] = frame_path

            # Update the progress bar
            pbar.update(1)

    #print(f"Removed {duplicate_frames} duplicate frames.")
    return duplicate_frames


def frames2text(num_frames):
    results = {}
    frame_directory = f"{os.path.splitext(args.filename)[0]}-frames"

    # Filter out duplicate frames
    num_unique_frames = num_frames - filter_duplicate_frames(frame_directory)
    frame_files = list(Path(frame_directory).glob('*'))

    # Convert frames to text
    if args.ocr_mode == "google_vision":
        with tqdm(total=len(frame_files), unit='frame', desc=f"Converting frames to text") as pbar:
            for video_frame in frame_files:
                # Load the frame image
                frame_path = f'{frame_directory}/{video_frame.name}'
                frame_image = Image.open(frame_path)

                # Perform OCR on the preprocessed image
                results[video_frame.name] = detect_text(frame_image)

                pbar.update(1)

    elif args.ocr_mode == "tesseract":
        with tqdm(total=len(frame_files), unit='frame', desc=f"Converting frames to text") as pbar:
            for video_frame in frame_files:
                results[video_frame.name] = pytesseract.image_to_string(Image.open(f'{frame_directory}/{video_frame.name}'),
                    lang='eng+osd',
                    config='--oem 1 --psm 1 --dpi 72',
                    timeout=2
                )
                pbar.update(1)
                
    elif args.ocr_mode == "easyocr":
        with tqdm(total=len(frame_files), unit='frame', desc=f"Converting frames to text") as pbar:
            for video_frame in frame_files:
                frame_path = f'{frame_directory}/{video_frame.name}'
                frame_image = Image.open(frame_path)
                results[video_frame.name] = reader.readtext(frame_image)[0] if reader.readtext(frame_image) else ''
                print(results)
                pbar.update(1)

    return results


def process_keywords(frame_content, frame_filename):
    for keyword in args.keywords[0].split(','):
        # Create directory to store result for this keyword
        keyword_result_directory = f"results/{os.path.splitext(args.filename.split('/')[-1])[0]}/{keyword}/frames"
        os.makedirs(keyword_result_directory, exist_ok=True)

        if fuzz.partial_ratio(frame_content, keyword) >= 70:
            # Copy frame file to result directory
            frame_file = f"{os.path.splitext(args.filename)[0]}-frames/{frame_filename}"
            if os.path.isfile(frame_file):
                shutil.copy2(frame_file, f"{keyword_result_directory}/{frame_filename}")

def frames2video(frames_pattern, output_video_path):
    frames = glob.glob(frames_pattern)
    frames.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Check if frames are available
    if not frames:
        # debug: print(f"No frames found for pattern: {frames_pattern}")
        return

    # Get the frame dimensions from the first frame
    first_frame = cv2.imread(frames[0])
    frame_height, frame_width, _ = first_frame.shape

    # Create a VideoWriter object to write the output video
    output_video = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.framerate,
        (frame_width, frame_height),
    )

    with tqdm(total=len(frames), unit='frame', desc="Converting frames to videos") as pbar:
        for frame_file in frames:
            frame = cv2.imread(frame_file)

            # Write the frame to the output video
            output_video.write(frame)
            pbar.update(1)

    # Release the VideoWriter object
    output_video.release()


def write_results(results):
    results_directory = os.path.splitext(args.filename.split("/")[-1])[0]  # Strip initial directory and remove extension
    results_directory_path = f"results/{results_directory}"
    os.makedirs(results_directory_path, exist_ok=True)

    if args.keywords:
        with tqdm(total=len(results), unit='frame', desc="Processing keywords") as pbar:
            for key in results:
                process_keywords(results[key], key)
                pbar.update(1)

    results_dict = {key: results[key] for key in results}

    results_file = f"{results_directory_path}/result.json"
    with open(results_file, 'w') as file:
        json.dump(results_dict, file, indent=4)

    if args.keywords:
        # Convert keyword frames back to a video
        for keyword in args.keywords[0].split(','):
            keyword_result_directory = f"{results_directory_path}/{keyword}"
            frames_pattern = f"{keyword_result_directory}/frames/frame*.jpg"
            output_video_path = f"{keyword_result_directory}/{keyword}.mp4"
            frames2video(frames_pattern, output_video_path)

    print(f"Results written to {results_file}")

if __name__ == "__main__":
    if args.mode == "usb":
        num_frames = frame_capture()
    elif args.mode == "video":
        num_frames = video2frames()
    results = frames2text(num_frames)
    write_results(results)
