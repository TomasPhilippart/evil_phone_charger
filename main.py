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
import numpy as np
from PIL import Image, ImageChops
import pytesseract
import os
import re
import cv2
import time
from moviepy.editor import VideoFileClip
import hashlib
from tqdm import tqdm


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="video", type=str, choices=['usb', 'video'], help='Either capture frames directly from USB device or use video file.')
parser.add_argument('--capture_time', default="100", type=int, help='How many seconds to capture. Unlimited is 0. (default: 10).')
parser.add_argument('--filename', required=True, type=str, help='Filename for video to process.')
parser.add_argument('--ocr_mode', default='tesseract', choices=['tesseract', 'google_vision'], help='Use either Tesseract (default) or Google Cloud Vision API (paid!)')
parser.add_argument('--framerate', default='20', type=int, choices=range(1, 31), help='Framerate for capture (default: 20).')
args = parser.parse_args()

# (OPTIONAL) Cleanup: remove garbage, easier for testing
#os.system('./cleanup.sh')

def frame_capture():
    args.filename = f"media/{args.filename}" # Change filename to include the media/ prefix, so we store frames there
    print("Starting frame capture... Press CTRL+C to stop recording frames.")
    
    # Create directory
    os.mkdir(f"{args.filename}-frames")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened(): 
        print("Unable to read camera feed")
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    i = 0
    time_start = time.time()
    try:
        while((args.capture_time != 0 and time.time() < time_start + args.capture_time) or (args.capture_time == 0)) :
            ret, frame = cap.read()
        
            if ret: 
                cv2.imwrite(f"{args.filename}-frames/frame{i}.jpg", frame)
                i+=1
        
            # Break the loop
            else:
                break 
        
            time.sleep(1/args.framerate)
    except KeyboardInterrupt:
        print("CTRL+C detected, stopping frame capture.")
        pass
    # When everything done, release the video capture and video write objects
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows() 

    return i


def video2frames():
    # load the video clip
    video_clip = VideoFileClip(args.filename)
    # make a folder by the name of the video file
    filename, _ = os.path.splitext(args.filename)
    filename += "-frames"
    if not os.path.isdir(filename):
        os.mkdir(filename)

    i = 0

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, args.framerate)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    for current_duration in tqdm(np.arange(0, video_clip.duration, step), desc="Converting video to frames", unit="frame"):
        # format the file name and save it
        frame_filename = os.path.join(filename, f"frame{i}.jpg") 
        i += 1 
        # save the frame with the current duration
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
                results[video_frame.name] = detect_text(f'{frame_directory}/{video_frame.name}')
                pbar.update(1)


    elif args.ocr_mode == "tesseract":
        with tqdm(total=len(frame_files), unit='frame', desc=f"Converting frames to text") as pbar:
            for video_frame in frame_files:
                try:
                    results[video_frame.name] = pytesseract.image_to_string(Image.open(f'{frame_directory}/{video_frame.name}'), lang='eng+osd', config='--oem 1 --psm 1 --dpi 72', timeout=2)
                except RuntimeError as timeout_error:
                    # Tesseract processing is terminated
                    pass
                pbar.update(1)
    return results
    
def write_results(results):
    results_file = os.path.splitext(args.filename.split("/")[-1])[0] # strip of initial directory (media/) and remove extension

    with open(f"results/{results_file}.txt", "w") as file:
        results = dict(sorted(results.items(), key=lambda x: int(re.findall(r'\d+', x[0])[0])))
        for key in results:
            file.write(f"Frame: '{key}'\n")
            file.write("------------\n")
            file.write(results[key])
            file.write("\n================================================\n")
    
    print(f"Results writen to results/{results_file}.txt.")        

if __name__ == "__main__":
    if args.mode == "usb":
        num_frames = frame_capture()
    elif args.mode == "video":
        num_frames = video2frames()
    results = frames2text(num_frames)
    write_results(results)