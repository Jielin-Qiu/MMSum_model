from utils import open_file, time_to_seconds
import os
import glob
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from transformers import RobertaTokenizer, RobertaModel

torch.set_num_threads(2)

# Load the pre-trained RoBERTa model and tokenizer
model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

list_of_annotations = glob.glob('../../jielin/msmo/annotation/*/*/*')

# Load the CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_dic = {}
test_dic = {}

count = 0

for annotation in tqdm(list_of_annotations, desc = 'Extracting text features: '):
    
    json_file = open_file(annotation)
    key = json_file['info']['video_id']
    keyframes = json_file['summary']
    summary_sequence = [item['summary'] for item in json_file['transcript']]

    # Tokenize the summary sequence and convert to IDs
    inputs = tokenizer(summary_sequence, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']

    # Pass the input IDs through the model to get the output embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        if embeddings.ndimension() ==1:
            embeddings = embeddings.unsqueeze(0).numpy()
        else:
            embeddings = embeddings.numpy()
        print(embeddings.shape)
    
    assert embeddings.shape[0] == len(summary_sequence)

    if key[-4:] == '0021' or key[-4:] == '0022' or key[-4:] == '0023' \
        or key[-4:] == '0024' or key[-4:] == '0025' or key[-4:] == '0026' \
        or key[-4:] == '0027' or key[-4:] == '0028' or key[-4:] == '0029':
            
        test_dic[f'{key}'] = embeddings
    else:
        train_dic[f'{key}'] = embeddings

    # count +=1 
    # if count == 50:
    #     break

np.save('summary_embeddings_roberta_base_train.npy', train_dic)
np.save('summary_embeddings_roberta_base_test.npy', test_dic)