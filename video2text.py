import helpers.video2frames as video2frames
#import frames2text
from pathlib import Path
import sys
from PIL import Image
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
video_file = sys.argv[1]
results = {}

# Cleanup
os.system('./cleanup.sh')

# Convert video to frames
print(f"Converting {video_file} to frames...")
video2frames.main(video_file)

# Convert frames to text (OCR)
print(f"Converting frames in directory {f'{video_file[:-4]}-frames'} to text...")
for video_frame in Path(f'{video_file[:-4]}-frames').glob('*'):
    try:
        results[video_frame.name] = pytesseract.image_to_string(Image.open(f'{video_file[:-4]}-frames/{video_frame.name}'), lang='eng', config='--psm 1', timeout=2)
    except RuntimeError as timeout_error:
        # Tesseract processing is terminated
        pass

# Writing to text file
with open(f"results/{video_file[:-4]}.txt", "w") as file:
    results = dict(sorted(results.items(), key=lambda x: int(re.findall(r'\d+', x[0])[0])))
    for key in results:
        file.write(f"Frame: '{key}'\n")
        file.write("------------\n")
        file.write(results[key])
        file.write("\n================================================\n")
        
    
print(f"Result saved in results/{video_file[:-4]}.txt")