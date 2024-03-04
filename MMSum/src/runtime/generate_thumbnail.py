import numpy as np

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import glob
from tqdm import tqdm
import random

import cv2
torch.set_num_threads(2)
import os
import json
import numpy as np
import h5py


def search_path_by_id(id, path_list):
    for path in path_list:
        if id in path:
            return path
    return None 



def open_file(path_to_file):
    file_extension = os.path.splitext(path_to_file)[1]
    file_reader = {
        '.json': lambda: json.load(open(path_to_file)),
        '.txt': lambda: open(path_to_file, 'r').read().split('\n'),
        '.npy': lambda: np.load(path_to_file, allow_pickle=True).item(),
        '.h5': lambda: h5py.File(path_to_file, 'r')
    }
    return file_reader.get(file_extension, lambda: None)()

def time_to_seconds(time_string):
    hours, minutes, seconds = map(int, time_string.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def extract_frame(cap, frame_num):
    # Set the frame position to the given frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Grab the next frame
    success = cap.grab()
    if not success:
        return None

    # Decode and return the grabbed frame
    _, frame = cap.retrieve()
    return frame


def extract_frames(video_file, start_time, end_time, num_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Could not open video file {video_file}")
        return []

    # Get the frame rate and total number of frames in the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the start and end frame numbers based on the given start and end times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Calculate the frame numbers for the key frames
    frame_nums = np.linspace(start_frame, end_frame, num_frames, dtype=np.int32)
    # Extract the key frames sequentially
    frames = []
    # frames2 = []
    for frame_num in frame_nums:
        # Set the frame position to the given frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the current frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the list of frames
        frames.append(frame_rgb)
        # frames2.append(frame_rgb.tolist())
    # Release the video capture object
    cap.release()

    return frames#, np.array(frames2)

path_to_videos = glob.glob('../../../../../jielin/msmo/video/*/*/*')
path_to_annotations = glob.glob('../../../../../jielin/msmo/annotation/*/*/*')
count = 0
available_fonts =glob.glob('fonts/*')
for i in tqdm(range(54)):
    
    results_whole_decoder = np.load(f'results/results_whole_{i}.npy', allow_pickle=True).item()
    sentences = results_whole_decoder['sentences']
    references = results_whole_decoder['references']
    selected_frames = results_whole_decoder['selected_frames']
    ids = results_whole_decoder['ids']
    count += len(ids)
    
    for j in range(len(sentences)):
        
        sentence = sentences[j]
        reference = references[j]
        selected_frame = selected_frames[j]
        id = ids[j]
        
        video = search_path_by_id(id, path_to_videos)
        annotation = search_path_by_id(id, path_to_annotations)
        # print(annotation)
        # print(sentence)
        # print(reference)
        # print(selected_frame)
        # print(id)
        # print(video)
        
        json_file = open_file(annotation)
        id = json_file['info']['video_id']
        keyframes = json_file['summary']
        
        start_time_seconds = 0
        end_time_seconds = time_to_seconds(json_file['info']['duration'])


        frames = extract_frames(video, start_time_seconds, end_time_seconds, 100)
        # frames = frames[:99]
        frame_indexed = frames[selected_frame]
        
        image = Image.fromarray(frame_indexed)
        image.save(f'run2/outputs2/{id}.png')
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        
        # Select a random font and size
        font_name = random.choice(available_fonts)
        font_size = random.randint(25, 200)

        # Specify the text content
        text = sentence

        # Get the font object with the randomly selected font and size
        font = ImageFont.truetype(font_name, size=font_size)

        # Get the size of the image
        image_width, image_height = image.size

        # Generate random position for the text
        text_width, text_height = draw.textsize(text, font=font)
        x = random.randint(0, abs(image_width - text_width))
        y = random.randint(0, abs(image_height - text_height))

        # Add the text to the image with random font, size, and position
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        # Save the modified image
        image.save(f"run2/outputs/{id}.png")