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
import cv2
import time
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, type=str, help='Filename for storage (stored in <filename>-frames/filename<i>.jpg).')
parser.add_argument('--framerate', default='20', type=int, choices=range(1, 60), help='Framerate for capture (default: 20).')
args = parser.parse_args()

# Create directory
os.mkdir(f"{args.filename}-frames")

# Create a VideoCapture object
cap = cv2.VideoCapture(1)
 
# Check if camera opened successfully
if not cap.isOpened(): 
	print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
  
i = 0
while True:
	ret, frame = cap.read()
  
	if ret: 
		cv2.imwrite(f"{args.filename}-frames/{args.filename}{i}.jpg", frame)
		i+=1
		# Press Q on keyboard to stop recording
		if cv2.waitKey(1) & 0xFF == ord('q'):
	  		break
 
  	# Break the loop
	else:
		break 
 
	time.sleep(1/args.framerate)
# When everything done, release the video capture and video write objects
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows() 