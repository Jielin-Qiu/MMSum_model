# coding=utf-8

# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
# that is licensed under Apache-2.0 license, i.e.

# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""


import copy
import math
import os
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import *

logger = logging.get_logger(__name__)


MMS_IG65M_EM_SIZE = 2048
MMS_S3D_EMB_SIZE = 512
MMS_VIT_EMB_SIZE = 2048
MMS_EFFNET_EMB_SIZE = 2048


# Based on  https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = (
            self.config.initializer_factor
        )  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
            )
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                input_ids.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1
            )
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class MMST5Stack(T5PreTrainedModel): #encoder
    def __init__(
        self,
        config,
        embed_tokens,
        num_video_enc_layers=None,
        use_video_ig65m=None,
        use_video_s3d=None,
        use_image_vit=None,
        use_image_effnet=None,
        use_image_self_attention=False,
    ):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.use_image_self_attention = use_image_self_attention

        # --MMS-- Text-Video Attention
        if not self.is_decoder:
            # Video encoder
            self.mms_video_positional_encoding = PositionalEncoding(
                emb_size=config.d_model, dropout=config.dropout_rate
            )
            mms_encoder_layer = nn.modules.transformer.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=8,
                dim_feedforward=2 * config.d_model,
                dropout=config.dropout_rate,
                batch_first=True,
            )
            self.mms_video_encoder = nn.modules.transformer.TransformerEncoder(
                encoder_layer=mms_encoder_layer,
                num_layers=num_video_enc_layers,
            )
            if self.use_image_self_attention:
                self.mms_image_self_attention = (
                    nn.modules.transformer.TransformerEncoder(
                        encoder_layer=mms_encoder_layer,
                        num_layers=num_video_enc_layers,
                    )
                )

            # Cross-modal attention
            self.mms_video_text_attention = nn.modules.activation.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True,
            )
            self.mms_sigmoid = nn.Sigmoid()

            self.mms_forget_gate = nn.modules.linear.Linear(
                in_features=2 * config.d_model, out_features=config.d_model, bias=False
            )
            self.mms_ff_forget_gate = nn.modules.linear.Linear(
                in_features=2 * config.d_model, out_features=config.d_model, bias=False
            )

            if use_video_ig65m:
                if use_video_s3d:
                    self.mms_video_rotate = nn.modules.linear.Linear(
                        in_features=MMS_IG65M_EM_SIZE + MMS_S3D_EMB_SIZE,
                        out_features=config.d_model,
                        bias=False,
                    )
                else:
                    self.mms_video_rotate = nn.modules.linear.Linear(
                        in_features=MMS_IG65M_EM_SIZE,
                        out_features=config.d_model,
                        bias=False,
                    )
            else:
                self.mms_video_rotate = nn.modules.linear.Linear(
                    in_features=MMS_S3D_EMB_SIZE,
                    out_features=config.d_model,
                    bias=False,
                )

            self.mms_image_source_attention = nn.modules.activation.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True,
            )

            self.mms_forget_gate_image = nn.modules.linear.Linear(
                in_features=2 * config.d_model, out_features=config.d_model, bias=False
            )
            self.mms_ff_forget_gate_image = nn.modules.linear.Linear(
                in_features=2 * config.d_model, out_features=config.d_model, bias=False
            )

            if use_image_vit:
                if use_image_effnet:
                    self.mms_image_rotate = nn.modules.linear.Linear(
                        in_features=MMS_VIT_EMB_SIZE + MMS_EFFNET_EMB_SIZE,
                        out_features=config.d_model,
                        bias=False,
                    )
                else:
                    self.mms_image_rotate = nn.modules.linear.Linear(
                        in_features=MMS_VIT_EMB_SIZE,
                        out_features=config.d_model,
                        bias=False,
                    )
            else:
                self.mms_image_rotate = nn.modules.linear.Linear(
                    in_features=MMS_EFFNET_EMB_SIZE,
                    out_features=config.d_model,
                    bias=False,
                )

            self.mms_image_selector_projection = nn.modules.linear.Linear(
                in_features=config.d_model,
                out_features=1,
                bias=False,
            )
        # --MMS-- Text-Video Attention

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
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
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs=hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        # print('hidden_states')
        # print(hidden_states.shape)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # --MMS-- Text-Video Attention
        if not self.is_decoder:
            if video_ig65m_emb is not None:
                
                if video_s3d_emb is not None:
                    video_features = torch.cat((video_ig65m_emb, video_s3d_emb), -1)
                else:
                    video_features = video_ig65m_emb
            else:
                video_features = video_s3d_emb

            # Project to the same dimension as text
            video_features = self.mms_video_rotate(video_features)

            # Encode with transformer model - Eq (3) in MLASK paper
            transformer_encoded_video = self.mms_video_encoder(
                src=self.mms_video_positional_encoding(video_features),
                src_key_padding_mask=video_padding_mask,
            )
            transformer_encoded_video = self.dropout(transformer_encoded_video)

            # Compute the text-queried visual features - same shape as hidden_states. Eq (9) in MLASK paper
            transformer_encoded_video, _ = self.mms_video_text_attention(
                query=hidden_states,
                key=transformer_encoded_video,
                value=transformer_encoded_video,
                key_padding_mask=video_padding_mask,
                need_weights=False,
            )

            # Use the forget gate - https://arxiv.org/pdf/2109.02401.pdf section 3.3
            transformer_encoded_video_sigmoid = self.mms_sigmoid(
                self.mms_forget_gate(
                    torch.cat((hidden_states, transformer_encoded_video), -1)
                )
            )
            transformer_encoded_video = transformer_encoded_video.mul(
                transformer_encoded_video_sigmoid
            )

            # Apply the forget gate, Eq (10) in MLASK paper
            hidden_states = self.mms_ff_forget_gate(
                torch.cat((hidden_states, transformer_encoded_video), -1)
            )
            hidden_states = self.dropout(hidden_states)

            # Frame selection part
            if image_vit_emb is not None:
                if image_effnet_emb is not None:
                    image_seq_emb = torch.cat((image_vit_emb, image_effnet_emb), -1)
                else:
                    image_seq_emb = image_vit_emb
            else:
                image_seq_emb = image_effnet_emb
            # Project to the same dimension as text
            original_image_seq_emb = self.mms_image_rotate(image_seq_emb)

            if self.use_image_self_attention:
                # Eq (5) in MLASK paper
                original_image_seq_emb = self.mms_image_self_attention(
                    src=self.mms_video_positional_encoding(original_image_seq_emb),
                    src_key_padding_mask=image_padding_mask,
                )

            encoder2image_mask = torch.ones_like(attention_mask)
            # Different masking compared to huggingface modules
            encoder2image_mask = (
                encoder2image_mask.bool()
                .masked_fill(attention_mask == 0, True)
                .masked_fill(attention_mask == 1, False)
            )

            # Compute attention to fused video+text representation
            image_seq_emb, _ = self.mms_image_source_attention(
                query=original_image_seq_emb,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=encoder2image_mask,
                need_weights=False,
            )

            # Use the forget gate - https://arxiv.org/pdf/2109.02401.pdf section 3.3
            image_seq_emb_sigmoid = self.mms_sigmoid(
                self.mms_forget_gate_image(
                    torch.cat((original_image_seq_emb, image_seq_emb), -1)
                )
            )
            image_seq_emb = image_seq_emb.mul(image_seq_emb_sigmoid)

            # $\widehat{V}_{frame}$, Section 4.3 from MLASK paper
            image_seq_emb = self.mms_ff_forget_gate_image(
                torch.cat((original_image_seq_emb, image_seq_emb), -1)
            )

            # B x Num_Img_Frames
            image_seq_emb_flattened = torch.squeeze(
                self.mms_image_selector_projection(image_seq_emb)
            )

        # --MMS-- Text-Video Attention

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    image_seq_emb_flattened,
                ]
                if v is not None
            )
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )


@add_start_docstrings(
    """T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING
)
class MMST5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(
        self,
        config,
        num_video_enc_layers: int = 2,
        use_video_ig65m: bool = True,
        use_video_s3d: bool = True,
        use_image_vit: bool = True,
        use_image_effnet: bool = True,
        smooth_cos_labels: bool = True,
        use_image_self_attention: bool = False,
    ):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MMST5Stack(
            encoder_config,
            self.shared,
            num_video_enc_layers,
            use_video_ig65m,
            use_video_s3d,
            use_image_vit,
            use_image_effnet,
            use_image_self_attention,
        )
        self.smooth_cos_labels = smooth_cos_labels
        self.use_image_self_attention = use_image_self_attention

        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        image_selection_loss = None

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                video_ig65m_emb=video_ig65m_emb,
                video_s3d_emb=video_s3d_emb,
                image_vit_emb=image_vit_emb,
                image_effnet_emb=image_effnet_emb,
                video_padding_mask=video_padding_mask,
                image_padding_mask=image_padding_mask,
            )

            # While training

            if tgt_img_cosine_scores is not None:
                image_seq_emb_flattened = encoder_outputs[-1]
                image_summary_loss = nn.BCEWithLogitsLoss(reduction="none")

                if image_vit_emb is not None:
                    vit_cosine_sim = self.cosine_sim(
                        image_vit_emb, torch.unsqueeze(tgt_image_vit_emb, 1)
                    )
                if image_effnet_emb is not None:
                    effnet_cosine_sim = self.cosine_sim(
                        image_effnet_emb, torch.unsqueeze(tgt_image_effnet_emb, 1)
                    )
                    if image_vit_emb is not None:
                        cosine_sim = (vit_cosine_sim + effnet_cosine_sim) / 2
                    else:
                        cosine_sim = effnet_cosine_sim
                else:
                    cosine_sim = vit_cosine_sim

                cosine_sim = torch.where(
                    cosine_sim > 0, cosine_sim, torch.zeros_like(cosine_sim)
                )
                tgt_img_cosine_scores = cosine_sim
                # Section 4.4 in MLASK paper, $C_{max}$ or $C_{smooth}$
                if not self.smooth_cos_labels:
                    tgt_image_labels = (
                        nn.functional.one_hot(
                            torch.argmax(tgt_img_cosine_scores, dim=1),
                            num_classes=tgt_img_cosine_scores.shape[1],
                        )
                        .float()
                        .view(-1)
                    )
                else:
                    tgt_image_labels = tgt_img_cosine_scores.view(-1)
                image_selection_loss = image_summary_loss(
                    input=image_seq_emb_flattened.view(-1),
                    target=tgt_img_cosine_scores.view(-1),
                )
                # Mask the loss - don't do this per batch but per flattened sequence
                image_selection_loss = torch.mean(
                    torch.where(
                        image_padding_mask.view(-1) == 0,
                        image_selection_loss,
                        torch.zeros_like(image_selection_loss),
                    )
                )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (
                (lm_logits,)
                + decoder_outputs
                + decoder_outputs[1:]
                + encoder_outputs
                + (image_seq_emb_flattened,)
            )
            if loss is not None:
                assert image_selection_loss is not None
                return (
                    loss,
                    image_selection_loss,
                ) + output
            else:
                return (output,)
        else:
            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past
