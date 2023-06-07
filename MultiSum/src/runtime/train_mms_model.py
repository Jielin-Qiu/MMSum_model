#!/usr/bin/env python

import pytorch_lightning as pl
import os
import sys

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


parser = argparse.ArgumentParser(description="MMS training parameters.")
parser.add_argument(
    "--start_with_text_frozen",
    type=int,
    default=0,
    help="Number of epochs with text encoder/decoder frozen",
)
parser.add_argument(
    "--mask_video_features",
    action="store_true",
    help="Whether to mask the video features during training/inference.",
)
parser.add_argument(
    "--use_video_ig65m",
    # default = None,
    action="store_true",
    help="Whether to use the video_ig65m features.",
)
parser.add_argument(
    "--use_video_s3d",
    # default = None,
    action="store_true",
    help="Whether to use the video_s3d features.",
)
parser.add_argument(
    "--use_image_vit",
    # default = None,
    action="store_true",
    help="Whether to use the image_vit features.",
)
parser.add_argument(
    "--use_image_effnet",
    # default = None,
    action="store_true",
    help="Whether to use the image_effnet features.",
)
parser.add_argument(
    "--smooth_cos_labels",
    # default = None,
    action="store_true",
    help="Whether to use the smooothed targets to train as opposed to a single closest target.",
)
parser.add_argument(
    "--use_pretrained_summarizer",
    action="store_true",
    help="Whether to use the model pre-trained on text summarization.",
)

parser.add_argument(
    "--version",
    type=int,
    default=2,
    help="Manual versioning, to be able to compute variance for several runs.",
)

mms_args = parser.parse_args()

training_name = (
    f"version={mms_args.version}_ep_txt_fr={mms_args.start_with_text_frozen}"
)

if mms_args.use_video_ig65m:
    training_name += "_v=ig65m"
if mms_args.use_video_s3d:
    training_name += "_v=s3d"
if mms_args.use_image_vit:
    training_name += "_i=vit"
if mms_args.use_image_effnet:
    training_name += "_i=eff"
if mms_args.smooth_cos_labels:
    training_name += "_smooth"
if mms_args.use_pretrained_summarizer:
    training_name += "_pretrain"
if mms_args.mask_video_features:
    training_name += "v_masked"


ROUGE_RAW_L_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{ROUGE_RAW_L_F:.2f}",
    # monitor="ROUGE_RAW_L_F",
    monitor = 'BLEU',
    mode="max",
    save_top_k=1,
)

# ROUGE_RAW_L_stop = EarlyStopping(monitor="ROUGE_RAW_L_F", mode="max", patience=5)

ROUGE_RAW_L_stop = EarlyStopping(monitor="BLEU", mode="max", patience=5)

# Section 6.3 in MLASK paper
summeCzech_ckpt = "__PATH_TO_mT5_FINE-TUNED_ON_SumeCzech_DATASET__"


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

train_loader = mms_data.train_dataloader()
val_loader = mms_data.val_dataloader()

tb_logger = TensorBoardLogger("trainings", name="mms_novinky_tb", version=training_name)
trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    logger=tb_logger,
    log_every_n_steps=50,
    val_check_interval=1.0,
    gradient_clip_val=5,
    accumulate_grad_batches=16,
    callbacks=[ROUGE_RAW_L_checkpoint, ROUGE_RAW_L_stop],
)

model = MultimodalTransformer(
    num_video_enc_layers=4,
    use_video_ig65m=mms_args.use_video_ig65m,
    use_video_s3d=mms_args.use_video_s3d,
    use_image_vit=mms_args.use_image_vit,
    use_image_effnet=mms_args.use_image_effnet,
    smooth_cos_labels=mms_args.smooth_cos_labels,
    lr_max_val=0.0005,
    lr_init_val=0,
    lr_warmup_steps=8000,
    pre_trained_summeczech_ckpt=summeCzech_ckpt
    if mms_args.use_pretrained_summarizer
    else "",
    start_with_text_frozen=mms_args.start_with_text_frozen,
    mask_video_features=mms_args.mask_video_features,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
