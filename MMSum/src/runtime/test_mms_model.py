#!/usr/bin/env python

import pytorch_lightning as pl

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
import os
_data_base = '../'

from model_mms import MultimodalTransformer
from data_laoder import MMSDataset, MMSDataModule
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer

import argparse
import numpy as np
import torch

torch.set_num_threads(2)


print(sys.argv)

# CKPT_PATH = './trainings/mms_novinky_tb/version=2_ep_txt_fr=0_v=ig65m_i=vit/checkpoints/epoch=0-step=834-ROUGE_RAW_L_F=0.08.ckpt' # seg
CKPT_PATH = './trainings/mms_novinky_tb/version=1_ep_txt_fr=0_v=ig65m_i=vit/checkpoints/epoch=4-step=559-ROUGE_RAW_L_F=1.65.ckpt' # whole
TEST_OR_VAL = 'val'

ROUGE_RAW_L_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{ROUGE_RAW_L_F:.2f}",
    monitor="ROUGE_RAW_L_F",
    mode="max",
    save_top_k=1,
)

ROUGE_RAW_L_stop = EarlyStopping(monitor="ROUGE_RAW_L_F", mode="max", patience=5)


mms_data = MMSDataModule(
    argparse.Namespace(
        articles_path=f"{_data_base}/data/",
        video_ig65m_path=f"{_data_base}/data/videos",
        # frames = f'{_data_base}/data/frames',
        # video_s3d_path=f"{_data_base}/video_mp4/s3d_how100m",
        video_s3d_path = None,
        img_extract_vit_path=f"{_data_base}/data/keyframes",
        img_tgt_vit_path=f"{_data_base}/data/thumbnails",
        # img_extract_eff_path=f"{_data_base}/video_mp4/efficientnet_b5",
        img_extract_eff_path = None,
        # img_tgt_eff_path=f"{_data_base}/image_jpeg/efficientnet_b5",
        img_tgt_eff_path = None,
        model_headline=False,
        max_src_len=1536,
        max_tgt_len=256,
        train_batch_size=2,
        val_batch_size=16,
        num_workers=16,
    )
)

if TEST_OR_VAL == "val":
    test_loader = mms_data.val_dataloader()
elif TEST_OR_VAL == "test":
    test_loader = mms_data.test_dataloader()
else:
    sys.exit(1)

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    log_every_n_steps=50,
    # max_steps = 1,
    val_check_interval=1.0,
    gradient_clip_val=5,
    accumulate_grad_batches=16,
    callbacks=[ROUGE_RAW_L_checkpoint, ROUGE_RAW_L_stop],
)

model = MultimodalTransformer.load_from_checkpoint(CKPT_PATH)

trainer.validate(model, dataloaders=test_loader, ckpt_path=CKPT_PATH)
