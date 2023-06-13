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
import helpers.video2frames as video2frames
from pathlib import Path
import numpy as np
from PIL import Image
import pytesseract
import os
import re
from moviepy.editor import VideoFileClip

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, type=str, help='Filename for video to process.')
parser.add_argument('--mode', default='tesseract', choices=['tesseract', 'google_vision'], help='Use either Tesseract (default) or Google Cloud Vision API (paid!)')
parser.add_argument('--framerate', default='20', type=int, choices=range(1, 60), help='Framerate for capture (default: 20).')
args = parser.parse_args()

# Cleanup
os.system('./cleanup.sh')

def video2frames():
    print("Converting video to frames...")
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
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_filename = os.path.join(filename, f"frame{i}.jpg") 
        i += 1 
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)


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

def frames2text():
    print(f"Converting frames to text using {args.mode} mode...")
    results = {}
    
    frame_directory = f"{os.path.splitext(args.filename)[0]}-frames"
    
    if args.mode == "google_vision":
        for video_frame in Path(frame_directory).glob('*'):
            results[video_frame.name] = detect_text(f'{frame_directory}/{video_frame.name}')

    elif args.mode == "tesseract":
        for video_frame in Path(frame_directory).glob('*'):
            try:
                results[video_frame.name] = pytesseract.image_to_string(Image.open(f'{frame_directory}/{video_frame.name}'), lang='eng', config='--psm 1', timeout=2)
            except RuntimeError as timeout_error:
                # Tesseract processing is terminated
                pass
    return results
    
def write_results(results):
    results_file = os.path.splitext(args.filename.split("/")[-1])[0] # strip of initial directory (media/) and remove extension

    print(f"Writing results to results/{results_file}.txt")    
    with open(f"results/{results_file}.txt", "w") as file:
        results = dict(sorted(results.items(), key=lambda x: int(re.findall(r'\d+', x[0])[0])))
        for key in results:
            file.write(f"Frame: '{key}'\n")
            file.write("------------\n")
            file.write(results[key])
            file.write("\n================================================\n")
        

    print(f"Result saved in results/{os.path.splitext(args.filename)[0]}.txt") 
    
    
if __name__ == "__main__":
    video2frames()
    results = frames2text()
    write_results(results)