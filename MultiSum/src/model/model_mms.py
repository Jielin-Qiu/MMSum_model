import json
import os
import unicodedata
import math
from typing import Union
import sys
from PIL import Image, ImageDraw, ImageFont
from statistics import mean
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import nltk
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from evaluate import load
from sacrebleu.metrics import BLEU
from rouge_raw import RougeRaw
from torchmetrics import RetrievalMAP, RetrievalRecall, RetrievalPrecision
from scipy.stats import pearsonr, kendalltau
# from nlgeval import NLGEval
from nltk.translate.meteor_score import meteor_score
from pycocotools.coco import COCO
from image_similarity_measures.quality_metrics import rmse, psnr, ssim, sre
from pycocoevalcap.eval import COCOEvalCap
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
spice_scorer = Spice()
cider_scorer = Cider()
bertscore = load('bertscore')
nltk.download('wordnet')
nltk.download('omw-1.4')
# nlg_eval = NLGEval()
torch.set_num_threads(2)

import re

def clean_text(text):
    # Remove non-alphanumeric characters
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    
    # Remove extra whitespaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    
    return cleaned_text

def parse_lists(predictions, references, parse_num):
    parsed_predictions = []
    parsed_references = []

    for pred, refs in zip(predictions, references):
        if len(pred) <= parse_num:
            parsed_predictions.append(pred)
            parsed_references.append(refs)
        else:
            pred_parts = [pred[i:i+parse_num] for i in range(0, len(pred), parse_num)]
            parsed_predictions.extend(pred_parts)
            parsed_references.extend([refs] * len(pred_parts))

    return parsed_predictions, parsed_references


from utils import generate_square_subsequent_mask
from mms_modeling_t5 import MMST5ForConditionalGeneration

IG65M_EM_SIZE = 512
S3D_EMB_SIZE = 512
VIT_EMB_SIZE = 768
EFFNET_EMB_SIZE = 2048
MAX_TGT_LEN = 250


class MultimodalTransformer(pl.LightningModule):
    def __init__(
        self,
        num_video_enc_layers: int,
        use_video_ig65m: bool = True,
        use_video_s3d: bool = True,
        use_image_vit: bool = True,
        use_image_effnet: bool = True,
        smooth_cos_labels: bool = False,
        lr_max_val: float = 0.0005,
        lr_init_val: float = 0,
        lr_warmup_steps: int = 4000,
        pre_trained_summeczech_ckpt: str = "",
        start_with_text_frozen=0,
        mask_video_features=False,
        use_image_self_attention=True,
    ):
        super().__init__()
        # Sanity checks
        assert use_video_ig65m or use_video_s3d
        assert use_image_vit or use_image_effnet
        self.save_hyperparameters()
        self.model = None
        self._create_model()

    def _create_model(self):
        # Used with pre-trained MT5 checkpoint
        if self.hparams.pre_trained_summeczech_ckpt != "":
            self.model = MMST5ForConditionalGeneration.from_pretrained(
                self.hparams.pre_trained_summeczech_ckpt,
                num_video_enc_layers=self.hparams.num_video_enc_layers,
                use_video_ig65m=self.hparams.use_video_ig65m,
                use_video_s3d=self.hparams.use_video_s3d,
                use_image_vit=self.hparams.use_image_vit,
                use_image_effnet=self.hparams.use_image_effnet,
                smooth_cos_labels=self.hparams.smooth_cos_labels,
                use_image_self_attention=self.hparams.use_image_self_attention,
            )
        else:
            self.model = MMST5ForConditionalGeneration.from_pretrained(
                "google/mt5-small",
                num_video_enc_layers=self.hparams.num_video_enc_layers,
                use_video_ig65m=self.hparams.use_video_ig65m,
                use_video_s3d=self.hparams.use_video_s3d,
                use_image_vit=self.hparams.use_image_vit,
                use_image_effnet=self.hparams.use_image_effnet,
                smooth_cos_labels=self.hparams.smooth_cos_labels,
                use_image_self_attention=self.hparams.use_image_self_attention,
            )

        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        # Evaluation metrics
        self.sacrebleu = BLEU()
        self.rouge = RougeRaw()

        self.rMAP = RetrievalMAP()
        self.rRec_1 = RetrievalRecall(k=1)
        self.rRec_5 = RetrievalRecall(k=5)
        self.rRec_10 = RetrievalRecall(k=10)
        
        self.rPre_1 = RetrievalPrecision(k=1)
        self.rPre_5 = RetrievalPrecision(k=5)
        self.rPre_10 = RetrievalPrecision(k=10)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        video_ig65m_emb=None,
        video_s3d_emb=None,
        image_vit_emb=None,
        image_effnet_emb=None,
        video_padding_mask=None,
        image_padding_mask=None,
        tgt_img_cosine_scores=None,
        tgt_image_vit_emb=None,
        tgt_image_effnet_emb=None,
    ):
        return self.model.forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            video_ig65m_emb,
            video_s3d_emb,
            image_vit_emb,
            image_effnet_emb,
            video_padding_mask,
            image_padding_mask,
            tgt_img_cosine_scores,
            tgt_image_vit_emb,
            tgt_image_effnet_emb,
        )

    def training_step(self, batch, batch_idx):
        # print(batch.keys())
        src_tokens = batch["src_ids"]
        src_padding_mask = batch["src_mask"]
        tgt_tokens = batch["tgt_ids"]
        tgt_padding_mask = batch["tgt_mask"]
        batch['src_img_cosine'] = []
        tgt_img_cosine_scores = batch["src_img_cosine"]
        # print('hello 0')

        video_padding_mask = batch["video_mask"]
        image_padding_mask = batch["src_img_mask"]

        # Video features
        if self.hparams.use_video_ig65m:
            video_ig65m_emb = batch["video_features_ig65m"]
        else:
            video_ig65m_emb = None
        if self.hparams.use_video_s3d:
            video_s3d_emb = batch["video_features_s3d"]
        else:
            video_s3d_emb = None

        # Image features
        if self.hparams.use_image_vit:
            image_vit_emb = batch["src_img_features_vit"]
            tgt_image_vit_emb = batch["tgt_img_features_vit"]
        else:
            image_vit_emb = None
            tgt_image_vit_emb = None
        if self.hparams.use_image_effnet:
            image_effnet_emb = batch["src_img_features_effnet"]
            tgt_image_effnet_emb = batch["tgt_img_features_effnet"]
        else:
            image_effnet_emb = None
            tgt_image_effnet_emb = None

        if self.hparams.mask_video_features:
            video_ig65m_emb = torch.randn_like(video_ig65m_emb)
            video_s3d_emb = torch.randn_like(video_s3d_emb)
        # print('hello 1 ')
        # Compute text summary loss
        text_summary_loss, image_selection_loss, *_ = self.forward(
            input_ids=src_tokens,
            attention_mask=src_padding_mask,
            decoder_attention_mask=tgt_padding_mask,
            labels=tgt_tokens,
            return_dict=False,
            video_ig65m_emb=video_ig65m_emb,
            video_s3d_emb=video_s3d_emb,
            image_vit_emb=image_vit_emb,
            image_effnet_emb=image_effnet_emb,
            video_padding_mask=video_padding_mask,
            image_padding_mask=image_padding_mask,
            tgt_img_cosine_scores=tgt_img_cosine_scores,
            tgt_image_vit_emb=tgt_image_vit_emb,
            tgt_image_effnet_emb=tgt_image_effnet_emb,
        )
        # print('hello 2 ')
        self.log(
            "img_loss",
            image_selection_loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
        )
        self.log(
            "summary_loss",
            text_summary_loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch),
        )

        return text_summary_loss + image_selection_loss

    def prediction_step(self, batch, batch_idx):
        src_tokens = batch["src_ids"]
        src_padding_mask = batch["src_mask"]
        video_padding_mask = batch["video_mask"]
        image_padding_mask = batch["src_img_mask"]

        # Video features
        if self.hparams.use_video_ig65m:
            video_ig65m_emb = batch["video_features_ig65m"]
            # print('video_ig65m')
            # print(video_ig65m_emb.shape)
        else:
            video_ig65m_emb = None
        if self.hparams.use_video_s3d:
            video_s3d_emb = batch["video_features_s3d"]
        else:
            video_s3d_emb = None

        # Image features
        if self.hparams.use_image_vit:
            image_vit_emb = batch["src_img_features_vit"]
            # print('image_vit_emb')
            # print(image_vit_emb.shape)
        else:
            image_vit_emb = None
        if self.hparams.use_image_effnet:
            image_effnet_emb = batch["src_img_features_effnet"]
        else:
            image_effnet_emb = None

        if self.hparams.mask_video_features:
            video_ig65m_emb = torch.randn_like(video_ig65m_emb)
            video_s3d_emb = torch.randn_like(video_s3d_emb)

        # Generate the summary using the text and video features
        txt_summary_tokens = self.model.generate(
            input_ids=src_tokens,
            attention_mask=src_padding_mask,
            video_ig65m_emb=video_ig65m_emb,
            video_s3d_emb=video_s3d_emb,
            video_padding_mask=video_padding_mask,
            image_effnet_emb=image_effnet_emb,
            image_padding_mask=image_padding_mask,
            image_vit_emb=image_vit_emb,
            num_beams=4,
            max_length=256,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )
        predicted_sent = self.tokenizer.batch_decode(
            txt_summary_tokens, skip_special_tokens=True
        )
        # tgt_tokens = batch["tgt_ids"]
        # tgt_padding_mask = batch["tgt_mask"]
        # batch['src_img_cosine'] = []
        # tgt_img_cosine_scores = batch["src_img_cosine"]
        # image_vit_emb = batch["src_img_features_vit"]
        # tgt_image_vit_emb = batch["tgt_img_features_vit"]
        # tgt_image_effnet_emb = None
        
        # _, _, *out1 = self.forward(
        #         input_ids=src_tokens,
        #         attention_mask=src_padding_mask,
        #         decoder_attention_mask=tgt_padding_mask,
        #         labels=tgt_tokens,
        #         return_dict=False,
        #         video_ig65m_emb=video_ig65m_emb,
        #         video_s3d_emb=video_s3d_emb,
        #         image_vit_emb=image_vit_emb,
        #         image_effnet_emb=image_effnet_emb,
        #         video_padding_mask=video_padding_mask,
        #         image_padding_mask=image_padding_mask,
        #         tgt_img_cosine_scores=tgt_img_cosine_scores,
        #         tgt_image_vit_emb=tgt_image_vit_emb,
        #         tgt_image_effnet_emb=tgt_image_effnet_emb,
        #     )

        # Use the encoder explicitly to obtain the per-frame scores
        *out2, per_frame_logits = self.model.encoder(
            input_ids=src_tokens,
            attention_mask=src_padding_mask,
            video_ig65m_emb=video_ig65m_emb,
            video_s3d_emb=video_s3d_emb,
            video_padding_mask=video_padding_mask,
            image_effnet_emb=image_effnet_emb,
            image_padding_mask=image_padding_mask,
            image_vit_emb=image_vit_emb,
            return_dict=False,
        )
        # print('src_img_Features')
        # print(image_vit_emb.shape)
        # print('out')
        # print(out1[0].shape)
        # print()
        # decoder_out = out1[1]
        # print(out1[1].shape)
        # print()
        # print(out2[0].shape)
        # print()
        # print(len(out1[2]))
        # print()
        # try:
        #     print(out1[2][0][0].shape)
        # except:
        #     print('out1[2][0] tuple')
        # # print()
        # try:
        #     print(out1[3].shape)
        # except:
        #     print('out1[3] tuple')
        # print()
        # try:
        #     print(out1[4].shape)
        # except:
        #     print('out1[4] tuple')
        
        

        per_frame_logits = per_frame_logits.masked_fill(image_padding_mask != 0.0, -1e6)

        return {"hyp": predicted_sent, "frame_scores": per_frame_logits, 
                'encoder_hidden_state': out2[0]}#, 'decoder_hidden_state' : decoder_out}

    def validation_step(self, batch, batch_idx):
        # print('hello 0 ')
        predictions = self.prediction_step(batch, batch_idx)
        # print('hello 1')
        if self.hparams.use_image_vit:
            # print(torch.unsqueeze(batch["tgt_img_features_vit"], 1).shape)
            vit_cosine_sim = self.cosine_sim(
                batch["src_img_features_vit"],
                # batch["tgt_img_features_vit"],
                torch.unsqueeze(batch["tgt_img_features_vit"], 1), # if training whole
            )
            # print(vit_cosine_sim)
        if self.hparams.use_image_effnet:
            effnet_cosine_sim = self.cosine_sim(
                batch["src_img_features_effnet"],
                torch.unsqueeze(batch["tgt_img_features_effnet"], 1),
            )
            if self.hparams.use_image_vit:
                cosine_sim = (vit_cosine_sim + effnet_cosine_sim) / 2
            else:
                cosine_sim = effnet_cosine_sim
        else:
            cosine_sim = vit_cosine_sim

        # print('hello 2')
        cosine_sim = torch.where(
            cosine_sim > 0, cosine_sim, torch.zeros_like(cosine_sim)
        ).cpu()

        cosine_sim_raw = cosine_sim.detach().clone()

        top_1_frame = torch.argmax(cosine_sim, dim=1)
        # unique_values = torch.unique(top_1_frame)
        # selected_frames = batch["video_features_ig65m"][:, unique_values, :]
        # print('unique_values')
        # print(top_1_frame)
        ids = batch['_id']
        hyp = predictions["hyp"],
        targ = batch["tgt"]
        # sent = [sent for sent in hyp]
        sent = hyp[0]
        refs = [sent for sent in targ]
        # print(len(sent))
        # print(len(refs))
        # print('hello 3')
        save_dic = {
            'sentences' : sent,
            'references' : refs,
            'selected_frames' : top_1_frame.tolist(),
            'ids'       : ids
        }
        
        np.save(f'results_whole_3/results_whole_{batch_idx}.npy', save_dic)
        # print('hello 4')
        
        # Open the image
        # image = Image.open("image.jpg")

        # # Create a drawing object
        # draw = ImageDraw.Draw(image)

        # # Specify the font and size
        # font = ImageFont.truetype("arial.ttf", size=24)

        # # Specify the text content and position
        # text = "Hello, World!"
        # position = (50, 50)

        # # Add the text to the image
        # draw.text(position, text, font=font, fill=(255, 255, 255))

        # # Save the modified image
        # image.save("image_with_text.jpg")
                

        top_1_frame_onehot = torch.nn.functional.one_hot(
            top_1_frame, num_classes=cosine_sim.shape[-1]
        )

        # We want to select frames based on threshold, but also make sure
        # that at least one frame (the most similar one, whatever the value) is chosen
        cosine_sim.index_add_(
            0,
            torch.arange(0, cosine_sim.shape[0]).type(torch.int64),
            top_1_frame_onehot.type(cosine_sim.type()),
        )

        mask = torch.ones(cosine_sim.size())
        above_threshold_9 = (cosine_sim >= 0.9) * mask
        above_threshold_75 = (cosine_sim >= 0.75) * mask

        indices = torch.repeat_interleave(
            torch.arange(0, cosine_sim.shape[0]).view(cosine_sim.shape[0], 1),
            cosine_sim.shape[1],
            dim=1,
        ).view(-1)
        
        # print('indices')
        # print(indices)
        # print('top_1_frame')
        # print(top_1_frame)
        # print('top_1_frame_onehot')
        # print(top_1_frame_onehot)
        # print('frame_Scores')
        # print(predictions["frame_scores"])
        # print('cosine')
        # print(cosine_sim)

        # Average cosine similarity between top1 frame and target
        cnn_cos_scores = []
        for ind, top1_ind in enumerate(
            np.array(
                torch.topk(predictions["frame_scores"].cpu(), dim=-1, k=1)[1].view(-1)
            )
        ):
            # print(predictions['frame_scores'].cpu().shape)
            if self.hparams.use_image_effnet:
                _cos1 = self.cosine_sim(
                    batch["tgt_img_features_effnet"][ind],
                    batch["src_img_features_effnet"][ind][top1_ind],
                )
            if self.hparams.use_image_vit:
                # print(batch['src_img_features_vit'].shape)
                # print(batch["src_img_features_vit"][ind][top1_ind].shape)
                # print(ind)
                # print(top1_ind)
                # print(batch['tgt_img_features_vit'].shape)
                # print(batch["src_img_features_vit"][ind][top1_ind])
                # print(batch["src_img_features_vit"].view(batch["src_img_features_vit"].shape[0], batch["src_img_features_vit"].shape[2]).shape)
                # try:
                _cos2 = self.cosine_sim(
                    batch["tgt_img_features_vit"][ind],
                    batch["src_img_features_vit"][ind][top1_ind],# for whole sentence
                    # batch["src_img_features_vit"][ind][0],
                )
                # print(rmse)
                # rsme = rmse(batch['tgt_img_features_vit'][ind].unsqueeze(), batch['src_img_features_vit'][ind])
                # rmse(org_img=np.random.rand(3,2,1), pred_img=np.random.rand(3,2,1))
                # except:
                #     _cos2 = 0
                # print(_cos2)
                if self.hparams.use_image_effnet:
                    cnn_cos_scores.append((_cos1 + _cos2) / 2)
                else:
                    cnn_cos_scores.append(_cos2)
            else:
                cnn_cos_scores.append(_cos1)

        # Correlation between model predictions and frame-scores
        # We consider only the non-masked values
        pearson_r_scores = []
        kendall_tau_scores = []
       
        # print(cosine_sim_raw.shape)
        # for ind in range(batch["src_img_cosine"].shape[0]):
        for ind in range(predictions['frame_scores'].shape[0]):
            _nonmasked_vals = (
                (predictions["frame_scores"][ind] == -1e6).nonzero().squeeze().view(-1)
            )
            # print('non_masked_vals')
            # print(_nonmasked_vals)
            
            if _nonmasked_vals.shape[0]:
                _nonmasked_ind = _nonmasked_vals[0]
                
                # print(' predictions["frame_scores"][ind][:_nonmasked_ind]')
                # print( predictions["frame_scores"][ind][:_nonmasked_ind])
                # print('cosine_sim_raw[ind][:_nonmasked_ind]')
                # print(cosine_sim_raw[ind][:_nonmasked_ind])
                
                try:
                    pearsonr_score = pearsonr(
                                        cosine_sim_raw[ind][:_nonmasked_ind].numpy(),
                                        predictions["frame_scores"][ind][:_nonmasked_ind]
                                        .cpu()
                                        .detach()
                                        .numpy(),
                                        )[0]
                    if math.isnan(pearsonr_score):
                        pearson_r_scores.append(0)
                    else:    
                        pearson_r_scores.append(pearsonr_score)
                except:
                    pearson_r_scores.append(0)
                kendall_tau_scores.append(
                    kendalltau(
                        cosine_sim_raw[ind][:_nonmasked_ind].numpy(),
                        predictions["frame_scores"][ind][:_nonmasked_ind]
                        .cpu()
                        .detach()
                        .numpy(),
                    )[0]
                )
            else:
                # try:
                    # print(cosine_sim_raw.shape)
                    # print(ind)
                    # print(predictions["frame_scores"].shape)
                    # print(pearsonr(
                    #         cosine_sim_raw[ind].numpy(),
                    #         # predictions["frame_scores"][ind].cpu().detach().numpy(),
                    #         predictions["frame_scores"].unsqueeze(1)[ind].cpu().detach().numpy(),
                    #     )[0])
                # print(cosine_sim_raw.shape)
                # print(cosine_sim_raw[ind].shape)
                # print(predictions["frame_scores"][ind].cpu().detach().numpy().shape)
                # print(predictions["frame_scores"].cpu().detach().numpy().shape)
                pearson_r_scores.append(
                    pearsonr(
                        # cosine_sim_raw[ind].numpy(),
                        cosine_sim_raw[ind].numpy(),
                        predictions["frame_scores"][ind].cpu().detach().numpy(),
                    )[0]
                )
                
                # except:
                #     pearson_r_scores.append(0)
                # try:
                kendall_tau_scores.append(
                    kendalltau(
                        cosine_sim_raw[ind].numpy(),
                        predictions["frame_scores"][ind].cpu().detach().numpy(),
                    )[0]
                )
                # except:
                    # kendall_tau_scores.append(0)

        # Mean average Precision, top1 and above threshold
        # print(predictions['frame_scores'].shape)
        # print(predictions['frame_scores'].view(-1).shape)
        # print(top_1_frame_onehot.view(-1).shape)
        # print(indices.shape)
        rMAP_top = self.rMAP(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rMAP_threshold_9 = self.rMAP(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_9.view(-1),
            indexes=indices,
        )
        rMAP_threshold_75 = self.rMAP(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_75.view(-1),
            indexes=indices,
        )
        rRec_1_top = self.rRec_1(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rRec_5_top = self.rRec_5(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rRec_10_top = self.rRec_10(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rPre_1_top = self.rPre_1(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rPre_5_top = self.rPre_5(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rPre_10_top = self.rPre_10(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=top_1_frame_onehot.view(-1),
            indexes=indices,
        )
        rRec_1_threshold_9 = self.rRec_1(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_9.view(-1),
            indexes=indices,
        )
        rRec_1_threshold_75 = self.rRec_1(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_75.view(-1),
            indexes=indices,
        )
        rRec_5_threshold_9 = self.rRec_5(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_9.view(-1),
            indexes=indices,
        )
        rRec_5_threshold_75 = self.rRec_5(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_75.view(-1),
            indexes=indices,
        )
        rRec_10_threshold_9 = self.rRec_10(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_9.view(-1),
            indexes=indices,
        )
        rRec_10_threshold_75 = self.rRec_10(
            preds=predictions["frame_scores"].cpu().view(-1),
            target=above_threshold_75.view(-1),
            indexes=indices,
        )
        
        # print(predictions['frame_scores'].unsqueeze(2).shape)
        # print(top_1_frame_onehot.unsqueeze(2).shape)
        
        # for i in range(len(batch['src_img_features_vit']))
        # batch["tgt_img_features_vit"][ind],
        # batch["src_img_features_vit"][ind][top1_ind],# for whole sentence
        # batch["src_img_features_vit"][ind][0],
        # print(batch["tgt_img_features_vit"].shape)
        rmse_scores = []
        psnr_scores = []
        ssim_scores = []
        sre_scores = []
        
        for i in range(batch['tgt_img_features_vit'].shape[0]):
            flattened_tensor = predictions['encoder_hidden_state'][i].flatten()
            subsampled_tensor = flattened_tensor[torch.randperm(flattened_tensor.numel())[:2048]]
            pred_img = subsampled_tensor.reshape(32, 64).unsqueeze(2)
            orig_img = batch['tgt_img_features_vit'][i].reshape(32, 64).unsqueeze(2)
            rmse_scores.append(rmse(org_img=orig_img.cpu().numpy(), pred_img=pred_img.cpu().numpy()))
            psnr_scores.append(psnr(org_img=orig_img.cpu().numpy(), pred_img=pred_img.cpu().numpy()))
            ssim_scores.append(ssim(org_img=orig_img.cpu().numpy(), pred_img=pred_img.cpu().numpy()))
            sre_scores.append(sre(org_img=orig_img.cpu().numpy(), pred_img=pred_img.cpu().numpy()))
            
        rmse_score = sum(rmse_scores)/len(rmse_scores)
        psnr_score = sum(psnr_scores)/len(psnr_scores)
        ssim_score = sum(ssim_scores)/len(ssim_scores)
        sre_score = sum(sre_scores)/len(sre_scores)
        
        
        
        
        return {
            "hyp": predictions["hyp"],
            "ref": batch["tgt"],
            "cnn_cos_scores": np.mean([_score.cpu() for _score in cnn_cos_scores]),
            "mAP_top": rMAP_top.numpy(),
            "mAP_threshold_0.9": rMAP_threshold_9.numpy(),
            "mAP_threshold_0.75": rMAP_threshold_75.numpy(),
            "Rec_1_top": rRec_1_top.numpy(),
            "Rec_5_top": rRec_5_top.numpy(),
            "Rec_10_top": rRec_10_top.numpy(),
            "Pre_1_top": rPre_1_top.numpy(),
            "Pre_5_top": rPre_5_top.numpy(),
            "Pre_10_top": rPre_10_top.numpy(),
            "Rec_1_threshold_0.9": rRec_1_threshold_9.numpy(),
            "Rec_5_threshold_0.75": rRec_5_threshold_75.numpy(),
            "Rec_10_threshold_0.9": rRec_10_threshold_9.numpy(),
            "Rec_1_threshold_0.75": rRec_1_threshold_75.numpy(),
            "Rec_5_threshold_0.9": rRec_5_threshold_9.numpy(),
            "Rec_10_threshold_0.75": rRec_10_threshold_75.numpy(),
            "Pearson_r": np.mean(pearson_r_scores),
            "Kendall_tau": np.mean(kendall_tau_scores),
            'RMSE_score' : rmse_score,
            'PSNR_score' : psnr_score,
            'SSIM_score' : ssim_score,
            'SRE_score'  : sre_score,
            'encoder_hidden_state' : predictions['encoder_hidden_state'],
            # 'decoder_hidden_state' : predictions['decoder_hidden_state']
        }

    def validation_epoch_end(self, outputs):
        # Frame selection evaluation
        cnn_cos_scores = np.mean([_r["cnn_cos_scores"] for _r in outputs])
        mAP_top = np.mean([_r["mAP_top"] for _r in outputs])
        mAP_threshold_9 = np.mean([_r["mAP_threshold_0.9"] for _r in outputs])
        mAP_threshold_75 = np.mean([_r["mAP_threshold_0.75"] for _r in outputs])
        Rec_1_top = np.mean([_r["Rec_1_top"] for _r in outputs])
        Rec_5_top = np.mean([_r["Rec_5_top"] for _r in outputs])
        Rec_10_top = np.mean([_r["Rec_10_top"] for _r in outputs])
        Pre_1_top = np.mean([_r["Pre_1_top"] for _r in outputs])
        Pre_5_top = np.mean([_r["Pre_5_top"] for _r in outputs])
        Pre_10_top = np.mean([_r["Pre_10_top"] for _r in outputs])
        Rec_1_threshold_9 = np.mean([_r["Rec_1_threshold_0.9"] for _r in outputs])
        Rec_1_threshold_75 = np.mean([_r["Rec_1_threshold_0.75"] for _r in outputs])
        Rec_5_threshold_9 = np.mean([_r["Rec_5_threshold_0.9"] for _r in outputs])
        Rec_5_threshold_75 = np.mean([_r["Rec_5_threshold_0.75"] for _r in outputs])
        Rec_10_threshold_9 = np.mean([_r["Rec_10_threshold_0.9"] for _r in outputs])
        Rec_10_threshold_75 = np.mean([_r["Rec_10_threshold_0.75"] for _r in outputs])
        Pearson_r = np.mean([_r["Pearson_r"] for _r in outputs])
        Kendall_tau = np.mean([_r["Kendall_tau"] for _r in outputs])
        PSNR = np.mean([_r["PSNR_score"] for _r in outputs])
        RMSE = np.mean([_r["RMSE_score"] for _r in outputs])
        SSIM = np.mean([_r["SSIM_score"] for _r in outputs])
        SRE = np.mean([_r["SRE_score"] for _r in outputs])
        
        
        self.log('PSNR', PSNR, on_step = False, on_epoch = True, sync_dist = True)
        self.log('RMSE', RMSE, on_step = False, on_epoch = True, sync_dist = True)
        self.log('SSIM', SSIM, on_step = False, on_epoch = True, sync_dist = True)
        self.log('SRE', SRE, on_step = False, on_epoch = True, sync_dist = True)

        self.log(
            "cnn_cos_scores",
            cnn_cos_scores,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("Pearson_r", Pearson_r, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "Kendall_tau", Kendall_tau, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("mAP_top", mAP_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "mAP_threshold_0.9",
            mAP_threshold_9,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "mAP_threshold_0.75",
            mAP_threshold_75,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("Rec_1_top", Rec_1_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Rec_5_top", Rec_5_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Rec_10_top", Rec_10_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Pre_1_top", Pre_1_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Pre_5_top", Pre_5_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Pre_10_top", Pre_10_top, on_step=False, on_epoch=True, sync_dist=True)
        self.log("F1_1_top", (2*Pre_1_top*Rec_1_top)/(Pre_1_top+Rec_1_top), on_step=False, on_epoch=True, sync_dist=True)
        self.log("F1_5_top", (2*Pre_5_top*Rec_5_top)/(Pre_5_top+Rec_5_top), on_step=False, on_epoch=True, sync_dist=True)
        self.log("F1_10_top", (2*Pre_10_top*Rec_10_top)/(Pre_10_top+Rec_10_top), on_step=False, on_epoch=True, sync_dist=True)

        self.log(
            "Rec_1_threshold_0.9",
            Rec_1_threshold_9,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Rec_1_threshold_0.75",
            Rec_1_threshold_75,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Rec_5_threshold_0.9",
            Rec_5_threshold_9,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Rec_5_threshold_0.75",
            Rec_5_threshold_75,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Rec_10_threshold_0.9",
            Rec_10_threshold_9,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Rec_10_threshold_0.75",
            Rec_10_threshold_75,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Text summarization evaluation
        predictions = [sent for _item in outputs for sent in _item["hyp"]]
        refs = [sent for _item in outputs for sent in _item["ref"]]
        
        # save_dic = {
        #     'predictions' : predictions,
        #     'refs'        : refs,
        #     'frames'      : [_r["encoder_hidden_state"] for _r in outputs]
        # }
        
        # np.save('results_whole_2.npy', save_dic)

        bleu_score = self.sacrebleu.corpus_score(predictions, refs).score
        rouge_score = self.rouge.corpus(refs, predictions)
        bert_score = bertscore.compute(predictions = predictions, references = refs, lang="en")
        refs = [[s] for s in refs]
        meteor = meteor_score(refs, predictions)
        b_p = mean(bert_score['precision'])
        b_f1 = mean(bert_score['f1'])
        b_r = mean(bert_score['recall'])
        
        self.log('BERTSCORE_F1', b_f1, on_step = False, on_epoch=True, sync_dist=True)
        self.log('BERTSCORE_P', b_p, on_step = False, on_epoch=True, sync_dist=True)
        self.log('BERTSCORE_R', b_r, on_step = False, on_epoch=True, sync_dist=True)
        
        self.log('METEOR', meteor, on_step = False, on_epoch=True, sync_dist=True)
        
        refs = {str(i): r for i, r in enumerate(refs)}
        preds = {str(i): [clean_text(p)] for i, p in enumerate(predictions)}
        # for i in preds.keys():
        #     s = preds[f'{i}'][0]
        #     r = refs[f'{i}'][0]
        #     # print(r)
        #     # print(len(r))
        #     print(s)
        #     print(len(s))
        # print(refs)
        # print()
        # print(preds)
        try:
            spice_score, _ = spice_scorer.compute_score(refs, preds)
        except:
            spice_score = 0
        
        cider_score, _ = cider_scorer.compute_score(refs, preds)
        
        self.log('SPICE', spice_score, on_step = False, on_epoch=True, sync_dist=True)
        self.log('CIDER', cider_score, on_step = False, on_epoch=True, sync_dist=True)
        
        self.log("BLEU", bleu_score, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "ROUGE_RAW_1_P",
            round(100 * rouge_score["1"].p, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_1_R",
            round(100 * rouge_score["1"].r, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_1_F",
            round(100 * rouge_score["1"].f, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "ROUGE_RAW_2_P",
            round(100 * rouge_score["2"].p, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_2_R",
            round(100 * rouge_score["2"].r, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_2_F",
            round(100 * rouge_score["2"].f, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "ROUGE_RAW_L_P",
            round(100 * rouge_score["L"].p, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_L_R",
            round(100 * rouge_score["L"].r, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "ROUGE_RAW_L_F",
            round(100 * rouge_score["L"].f, 2),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Enable different learning rate for pre-trained weights
        model_params = list(self.model.named_parameters())

        def is_mms(n):
            return "mms_" in n

        grouped_parameters = [
            {"params": [p for n, p in model_params if is_mms(n)]},
            {"params": [p for n, p in model_params if not is_mms(n)]},
        ]

        return torch.optim.Adam(
            grouped_parameters,
            lr=self.hparams.lr_init_val,
            betas=(0.9, 0.98),
            eps=1e-09,
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)
        # Linear increase for the first warmup_steps, then inverse square root decrease
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr = self.hparams.lr_init_val + self.trainer.global_step * (
                (self.hparams.lr_max_val - self.hparams.lr_init_val)
                / self.hparams.lr_warmup_steps
            )
        else:
            lr = (
                self.hparams.lr_max_val
                * self.hparams.lr_warmup_steps**0.5
                * self.trainer.global_step**-0.5
            )
        lr_mms = lr
        lr_text = lr

        if (
            self.hparams.start_with_text_frozen
            and self.trainer.current_epoch < self.hparams.start_with_text_frozen
        ):
            lr_text = 0.0

        # MMS params
        optimizer.param_groups[0]["lr"] = lr_mms
        # Pre-trained text params
        optimizer.param_groups[1]["lr"] = lr_text

        self.log("learning_rate_mms", lr_mms, on_step=True, on_epoch=False)

        self.log("learning_rate_text", lr_text, on_step=True, on_epoch=False)
