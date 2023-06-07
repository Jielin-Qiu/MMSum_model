import torch

from typing import List, Union
from torch import Tensor
import json
import os
import numpy as np
import h5py

def open_file(path_to_file):
    file_extension = os.path.splitext(path_to_file)[1]
    file_reader = {
        '.json': lambda: json.load(open(path_to_file)),
        '.txt': lambda: open(path_to_file, 'r').read().split('\n'),
        '.npy': lambda: np.load(path_to_file, allow_pickle=True).item(),
        '.h5': lambda: h5py.File(path_to_file, 'r')
    }
    return file_reader.get(file_extension, lambda: None)()

def split_list(mylist: List, chunk_size: Union[int]):
    """
    Splits list into list of lists of given size. The last chunk may be of different size.
    """
    return [
        mylist[offs : offs + chunk_size] for offs in range(0, len(mylist), chunk_size)
    ]


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[:, 0] = decoder_start_token_id

    return shifted_input_ids
