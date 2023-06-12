import helpers.video2frames as video2frames
from pathlib import Path
import sys
from PIL import Image
import pytesseract
import helpers.frame2text_google as frame2text_google
import time
import os
import re


pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
video_file = sys.argv[1]

results = {}

# Cleanup
os.system('./cleanup.sh')

# Warning for google cloud vision api billing
print("WARNING! Using Google Cloud Vision API - this will cost money... you have 5 seconds to cancel.")
time.sleep(5)

# Convert video to frames
print(f"Converting {video_file} to frames...")
video2frames.main(video_file)

# Convert frames to text (OCR)
print(f"Converting frames in directory {f'{video_file[:-4]}-frames'} to text...")
with open(f"results/{video_file[:-4]}.txt", "w") as file:
    for video_frame in Path(f'{video_file[:-4]}-frames').glob('*'):
        results[video_frame.name] = frame2text_google.detect_text(f'{video_file[:-4]}-frames/{video_frame.name}')
        
        
# Writing to text file
with open(f"results/{video_file[:-4]}.txt", "w") as file:
    results = dict(sorted(results.items(), key=lambda x: int(re.findall(r'\d+', x[0])[0])))
    for key in results:
        file.write(f"Frame: '{key}'\n")
        file.write("------------\n")
        file.write(results[key])
        file.write("\n================================================\n")
        
    
print(f"Result saved in results/{video_file[:-4]}.txt")