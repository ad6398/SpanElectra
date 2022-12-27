# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

##ad look if we require license or not

""" spanElectra <3 model configuration """

import logging

from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SPANELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = (
    {}  # "add downlaodable link of all configs here"
)


class SpanElectraConfig(PretrainedConfig):
    # r"""TBA"""
    pretrained_config_archive_map = SPANELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "spanelectra"

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=2,
        max_span_len=20,
        position_embedding_size=200,
        target_layer=-1,
        dummy_id=0,
        use_SBO=False,
        all_token_clf=True,
        ignore_label=2,
        max_seq_len=512,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_size = position_embedding_size
        self.target_layer = target_layer
        self.max_span_len = max_span_len
        self.dummy_id = dummy_id
        self.use_SBO = use_SBO
        self.all_token_clf = all_token_clf
        self.ignore_label = ignore_label
        self.max_seq_len = max_seq_len
