"""
Instructions: 

Before training, the raw video files must be converted into frame sequences.

The following module performs the following tasks:
1. Use OpenCV to uniformly sample frames from the video files, using the `get_frames` function.
2. Write the extracted frames as JPEG images to the specified directory using the `store_frames` function.
          store_path (str): Directory path where the frames will be saved.
"""
import os
import cv2
import numpy as np

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn

from utils import get_frames, store_frames  # Main functions to be used in the script


root = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
video_dir = os.path.join(root, "UCF50_video")  # Directory containing video files
store_path = os.path.join(root, "UCF50")  # Directory to store the frames

def preprocess_video(store_path):
    """
    Preprocess video files by extracting frames and saving them as JPEG images.
    
    This function iterates through all video files in the specified directory,
    extracts frames using OpenCV, and saves them to a designated directory.
    """
    # video_dir = "video_action\UCF50_videos"  # Directory containing video files
    root = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    video_dir = os.path.join(root, "UCF50_video")  # Directory containing video files
    if not os.path.exists(store_path):
        os.makedirs(store_path)  # Create the directory if it doesn't exist

    for dir in os.listdir(video_dir):
        # Iterate through all folders
        print(dir)
        for filename in os.listdir(os.path.join(video_dir, dir)):
            if filename.endswith(".avi"):
                print(filename)
                vid_path = os.path.join(video_dir, dir, filename)
                frames, v_len = get_frames(vid_path, n_frames=16)  # Extract 16 frames from the video
                store_path_spec = os.path.join(store_path, dir, filename.split('.')[0])
                os.makedirs(store_path_spec, exist_ok=True)
                store_frames(frames, store_path_spec)  # Save the extracted frames as JPEG images

# Make callable
if __name__ == "__main__":
    store_path = os.path.join(root, "UCF50")  # Directory to store the frames
    preprocess_video(store_path)  # Call the preprocessing function to start the process
    print(f"Frames extracted and stored in {store_path}")