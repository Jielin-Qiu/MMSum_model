import csv
import json
import os
import unicodedata

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from utils import generate_square_subsequent_mask


class MMSDataset(Dataset):
    """
    Dataloder used to process the MLASK dataset
    """

    def __init__(self, args, mode):
        self.args = args
        assert mode in ["dev", "test", "train"]
        self.mode = mode
        self.ids = None

        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        self._read_articles()
        self._read_videos()
        self._read_images()
        print(self.use_ig65m)
        print(self.use_s3d_how100m)
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        _return_dict = {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
            "_id": self._ids[idx],
        }

        if self.use_ig65m:
            _return_dict["video_features_ig65m"] = np.load(self.videos[idx]["ig65m"])
        if self.use_s3d_how100m:
            _return_dict["video_features_s3d"] = np.load(self.videos[idx]["s3d"])

        if self.use_vit:
            _return_dict["src_img_features_vit"] = np.load(self.src_imgs[idx]["vit"])
            _return_dict["tgt_img_features_vit"] = np.load(self.tgt_imgs[idx]["vit"])
        if self.use_effnet:
            _return_dict["src_img_features_effnet"] = np.load(
                self.src_imgs[idx]["effnet"]
            )
            _return_dict["tgt_img_features_effnet"] = np.load(
                self.tgt_imgs[idx]["effnet"]
            )

        return _return_dict

    def _read_videos(self):
        """
        Reads the video features
        """
        self.use_ig65m = self.args.video_ig65m_path is not None
        self.use_s3d_how100m = self.args.video_s3d_path is not None

        # At least one video feature must be used
        assert self.use_ig65m or self.use_s3d_how100m
        # This function should be called only after reading textual data
        assert self.ids is not None

        self.videos = []
        self._ids = []
        for _id in self.ids:
            _video_paths = {}
            _video_dir = _id
            if self.use_ig65m:
                _video_paths["ig65m"] = os.path.join(
                    self.args.video_ig65m_path, _video_dir + ".npy"
                )
                
            if self.use_s3d_how100m:
                _video_paths["s3d"] = os.path.join(
                    self.args.video_s3d_path, _video_dir + ".npy"
                )
            self.videos.append(_video_paths)
            self._ids.append(str(_id))

    def _read_images(self):
        """
        Reads the image features
        """
        self.use_vit = (
            self.args.img_extract_vit_path is not None
            and self.args.img_tgt_vit_path is not None
        )
        self.use_effnet = (
            self.args.img_extract_eff_path is not None
            and self.args.img_tgt_eff_path is not None
        )

        # At least one image feature must be used
        assert self.use_vit or self.use_effnet
        # This function should be called only after reading textual data
        assert self.ids is not None

        self.src_imgs = []
        self.tgt_imgs = []
        for _id in self.ids:
            _src_img_paths = {}
            _tgt_img_paths = {}
            # All the data instances are stored in a simple tree-like structure
            _img_dir = _id
            if self.use_vit:
                _src_img_paths["vit"] = os.path.join(
                    self.args.img_extract_vit_path, _img_dir + ".npy"
                )
                _tgt_img_paths["vit"] = os.path.join(
                    self.args.img_tgt_vit_path, _img_dir + ".npy"
                )
            if self.use_effnet:
                _src_img_paths["effnet"] = os.path.join(
                    self.args.img_extract_eff_path, _img_dir + ".npy"
                )
                _tgt_img_paths["effnet"] = os.path.join(
                    self.args.img_tgt_eff_path, _img_dir + ".npy"
                )
            self.src_imgs.append(_src_img_paths)
            self.tgt_imgs.append(_tgt_img_paths)

    def _read_articles(self):
        """
        Read textual documents
        """
        path = self.args.articles_path
        model_headline = self.args.model_headline

        df = pd.read_csv(
            os.path.join(path, f"{self.mode}_mms_joined.tsv"),
            sep="\t",
            quoting=csv.QUOTE_NONE,
        )

        df.columns = ["id", "date", "headline", "article", "abstract"]

        self.ids = df.id.values
        self.src = df.article.values
        if model_headline:
            self.tgt = df.headline.values
        else:
            self.tgt = df.abstract.values

    def collate_fn(self, batch):
        max_src_len = self.args.max_src_len
        max_tgt_len = self.args.max_tgt_len
        
        # Source tokens
        src_encoded = self.tokenizer(
            [_item["src"] for _item in batch],
            padding="longest",
            truncation=True,
            max_length=max_src_len,
        )

        src_ids = torch.tensor(src_encoded["input_ids"])
        src_mask = torch.tensor(src_encoded["attention_mask"])

        # Target tokens
        tgt_encoded = self.tokenizer(
            [_item["tgt"] for _item in batch],
            padding="longest",
            truncation=True,
            max_length=max_tgt_len,
        )

        tgt_ids = torch.tensor(tgt_encoded["input_ids"])
        tgt_mask = torch.tensor(tgt_encoded["attention_mask"])

        _return_dict = {
            "src": [_item["src"] for _item in batch],
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt": [_item["tgt"] for _item in batch],
            "tgt_ids": tgt_ids,
            "tgt_mask": tgt_mask,
            "_id": [_item["_id"] for _item in batch],
        }

        # Video features, maximal length is taken care of during feature extraction
        if self.use_s3d_how100m:
            video_features_s3d = np.zeros(
                [
                    len(batch),
                    max([_item["video_features_s3d"].shape[0] for _item in batch]),
                    batch[0]["video_features_s3d"].shape[-1],
                ]
            )
            video_mask_s3d = np.full(video_features_s3d.shape[:2], float("-inf"))
        if self.use_ig65m:
            video_features_ig65m = np.zeros(
                [
                    len(batch),
                    max([_item["video_features_ig65m"].shape[0] for _item in batch]),
                    batch[0]["video_features_ig65m"].shape[-1],
                ]
            )
            video_mask_ig65m = np.full(video_features_ig65m.shape[:2], float("-inf"))

        for _iter, _item in enumerate(batch):
            if self.use_s3d_how100m:
                video_features_s3d[_iter][
                    : _item["video_features_s3d"].shape[0],
                    : _item["video_features_s3d"].shape[1],
                ] = _item["video_features_s3d"]
                video_mask_s3d[_iter][: _item["video_features_s3d"].shape[0]] = 0.0
            if self.use_ig65m:
                video_features_ig65m[_iter][
                    : _item["video_features_ig65m"].shape[0],
                    : _item["video_features_ig65m"].shape[1],
                ] = _item["video_features_ig65m"]
                video_mask_ig65m[_iter][: _item["video_features_ig65m"].shape[0]] = 0.0

        if self.use_s3d_how100m and self.use_ig65m:
            assert np.array_equal(video_mask_s3d, video_mask_ig65m)

        if self.use_s3d_how100m:
            _return_dict["video_features_s3d"] = torch.tensor(
                video_features_s3d
            ).float()
            _return_dict["video_mask"] = torch.tensor(video_mask_s3d)
        if self.use_ig65m:
            _return_dict["video_features_ig65m"] = torch.tensor(
                video_features_ig65m
            ).float()
            _return_dict["video_mask"] = torch.tensor(video_mask_ig65m)

        # Image features extracted from source video - we do extract a single feature for every 1s of video, up to 300 features
        if self.use_vit:
            src_img_features_vit = np.zeros(
                [
                    len(batch),
                    max([_item["src_img_features_vit"].shape[0] for _item in batch]),
                    batch[0]["src_img_features_vit"].shape[-1],
                ]
            )
            src_img_mask_vit = np.full(src_img_features_vit.shape[:2], 1.0)
            tgt_img_features_vit = np.stack(
                [_item["tgt_img_features_vit"][0] for _item in batch]
            )
        if self.use_effnet:
            src_img_features_effnet = np.zeros(
                [
                    len(batch),
                    max([_item["src_img_features_effnet"].shape[0] for _item in batch]),
                    batch[0]["src_img_features_effnet"].shape[-1],
                ]
            )
            src_img_mask_effnet = np.full(src_img_features_effnet.shape[:2], 1.0)
            tgt_img_features_effnet = np.stack(
                [_item["tgt_img_features_effnet"][0] for _item in batch]
            )

        for _iter, _item in enumerate(batch):
            if self.use_vit:
                src_img_features_vit[_iter][
                    : _item["src_img_features_vit"].shape[0],
                    : _item["src_img_features_vit"].shape[1],
                ] = _item["src_img_features_vit"]
                src_img_mask_vit[_iter][: _item["src_img_features_vit"].shape[0]] = 0.0
            if self.use_effnet:
                src_img_features_effnet[_iter][
                    : _item["src_img_features_effnet"].shape[0],
                    : _item["src_img_features_effnet"].shape[1],
                ] = _item["src_img_features_effnet"]
                src_img_mask_effnet[_iter][
                    : _item["src_img_features_effnet"].shape[0]
                ] = 0.0

        if self.use_s3d_how100m and self.use_ig65m:
            assert np.array_equal(src_img_mask_vit, src_img_mask_effnet)

        if self.use_vit:
            _return_dict["src_img_features_vit"] = torch.tensor(
                src_img_features_vit
            ).float()
            _return_dict["src_img_mask"] = torch.tensor(src_img_mask_vit)
            _return_dict["tgt_img_features_vit"] = torch.tensor(
                tgt_img_features_vit
            ).float()
        if self.use_effnet:
            _return_dict["src_img_features_effnet"] = torch.tensor(
                src_img_features_effnet
            ).float()
            _return_dict["src_img_mask"] = torch.tensor(src_img_mask_effnet)
            _return_dict["tgt_img_features_effnet"] = torch.tensor(
                tgt_img_features_effnet
            ).float()
        return _return_dict


class MMSDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        train_set = MMSDataset(args, "train")
        val_set = MMSDataset(args, "dev")
        test_set = MMSDataset(args, "test")
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=train_set.collate_fn,
        )
        self.val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=val_set.collate_fn,
        )
        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=test_set.collate_fn,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
