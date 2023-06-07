import os
import json
import openai
import tiktoken
import time
import backoff
import random
import h5py
import numpy as np
import cv2
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def check_video_file(video_file):
    # Check if file exists
    if not os.path.isfile(video_file):
        print("File does not exist")
        return False

    # Check if file is a valid video file
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Could not open video file")
            return False
        else:
            print("Video file opened successfully")

        # Check the encoding format of the video file
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        encoding_format = chr(fourcc & 0xFF) + chr((fourcc & 0xFF00) >> 8) + chr((fourcc & 0xFF0000) >> 16) + chr((fourcc & 0xFF000000) >> 24)
        print("Video file encoding format:", encoding_format)

        # Check for the required codecs
        codec = cv2.VideoWriter_fourcc(*'XVID')
        if not cv2.VideoWriter_fourcc(*encoding_format) == codec:
            print("Required codec not found")
            return False

        # Check if the file is corrupted or incomplete
        ret, frame = cap.read()
        if not ret:
            print("Video file is corrupted or incomplete")
            return False

        # Release the video capture object
        cap.release()
    except:
        print("An error occurred while checking the video file")
        return False

    return True

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



def time_to_seconds(time_string):
    hours, minutes, seconds = map(int, time_string.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def open_file(path_to_file):
    file_extension = os.path.splitext(path_to_file)[1]
    file_reader = {
        '.json': lambda: json.load(open(path_to_file)),
        '.txt': lambda: open(path_to_file, 'r').read().split('\n'),
        '.npy': lambda: np.load(path_to_file, allow_pickle=True).item(),
        '.h5': lambda: h5py.File(path_to_file, 'r')
    }
    return file_reader.get(file_extension, lambda: None)()




def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,openai.error.APIError),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay =1
                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)