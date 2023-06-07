from utils import open_file, time_to_seconds, extract_frames
import os
import glob
import torch
import torchvision.transforms as transforms
import clip
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

torch.set_num_threads(2)

list_of_keyframes = glob.glob('../../jielin/msmo/keyframe/*/*/*')

# Load the CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Modify the model to output features of size 2048
model.visual.output_dim = 2048

# Define the transform to preprocess the input frames
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    preprocess
])

class VideoFramesDataset(Dataset):
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        preprocessed_frame = self.transform(Image.fromarray(frame))
        return preprocessed_frame

linear = torch.nn.Linear(512, 2048, dtype=torch.float16).to(device)
model.to(device)

save_np_dic = {}

batch_size = 128

corrupted_videos = []

count = 0

for path in tqdm(list_of_keyframes, desc = 'Extracting features ...'):
    id = path.split('/')[-1]
    image_data = []

    for image in os.listdir(path):
        image = Image.open(os.path.join(path, image))
        image_array = np.array(image)
        image_data.append(image_array)

    stacked_array = np.stack(image_data, axis = 0)

    dataset = VideoFramesDataset(stacked_array, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    features_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = model.encode_image(batch)
            features = linear(features.to(device))
            features_list.append(features.cpu())

    features = torch.cat(features_list, dim=0)
    save_np_dic[f'{id}'] = features.numpy()

    # count +=1 
    
    # if count == 50:
    #     break
    # print(save_np_dic)

# The features tensor has shape [num_frames, feature_size]
np.save('msmo_clip_summ_features.npy', save_np_dic)