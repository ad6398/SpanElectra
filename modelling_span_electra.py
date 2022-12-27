import logging
import os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers.activations import get_activation
from transformers.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertLayerNorm,
    BertPooler,
    BertPreTrainedModel,
)

from utilis import (
    get_f1,
    get_mlm_loss_out_sbo_labels,
    ceLoss,
    get_pre_from_span_level_logits,
    get_pre,
    get_disc_in_disc_labels,
    get_flat_acc,
)

# from transformers import SapnElectraConfig, add_start_docstrings
# from .file_utils import add_start_docstrings_to_callable

logger = logging.getLogger(__name__)


SPANELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP = {  # ad "add downlaodable link of all models here"
    # model_name : model_link, for eg
    # "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/pytorch_model.bin"
}


def load_tf_weights_in_electra(
    model, config, tf_checkpoint_path, discriminator_or_generator="discriminator"
):
    pass  # ad to be defined later


class SpanElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # CD

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SpanElectraPretrainedModels(BertPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    # config_class = SapnElectraConfig
    pretrained_model_archive_map = SPANELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_electra  ##ad
    base_model_prefix = "spanelectra"

    def get_extended_attention_mask(self, attention_mask, input_shape, device):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, head_mask):
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        num_hidden_layers = self.config.num_hidden_layers
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


class SpanElectraModel(SpanElectraPretrainedModels):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = SpanElectraEmbeddings(config)
        ##ad in case hidden size and embedding size is not same
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )

        self.encoder = BertEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    #     @add_start_docstrings_to_callable(SPANELECTRA_INPUTS_DOCSTRING)  ##ad why this error in indent/

    @classmethod
    def load_from_pretrained(
        cls, config, weight_path, modelType="disc", is_check_point=False
    ):
        """
        load model from pretrained part
        weight_path: path to loaded
        modelType: gen for generator, disc for discriminator
        """
        state_dict = torch.load(weight_path)
        if is_check_point:
            state_dict = state_dict["model_state_dict"]

        model = cls(config=config)
        for key in state_dict.keys():
            if "embeddings" == str(key) or modelType + "_embed" == str(key):
                model.embeddings.weight.copy_(state_dict[key])

            if modelType + "_embeddings_project" == str(key):
                model.embeddings_project.weight.copy_(state_dict[key])

            if "encoder" == str(key) or modelType + "_encoder" == str(key):
                model.encoder.weight.copy_(state_dict[key])

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        head_mask = self.get_head_mask(head_mask)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)
        ##ad return of encoder is  last-layer hidden state, (all hidden states if out_hs =True), (all attentions if out_attn== True)
        hidden_states = self.encoder(
            hidden_states, attention_mask=extended_attention_mask, head_mask=head_mask
        )

        return hidden_states


class MLPWithLayerNorm(nn.Module):  ##diff
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = get_activation(self.config.hidden_act)
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = get_activation(self.config.hidden_act)
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden):
        return self.layer_norm2(
            self.non_lin2(
                self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))
            )
        )


class SpanElectraGeneratorPredictionHead(nn.Module):
    def __init__(
        self,
        config,
        bert_model_embedding_weights,
        max_span_len=20,
        position_embedding_size=200,
    ):
        super().__init__()
        self.config = config
        self.max_span_len = max_span_len
        self.position_embeddings = nn.Embedding(max_span_len, position_embedding_size)

        # max_span_len = max word pieces b/w a pair
        self.mlp_layer_norm = MLPWithLayerNorm(
            config, config.hidden_size * 2 + position_embedding_size
        )
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        ##ad spanBert assumes embedding size is same as hidden_size so you have to insert embedding project stuff too
        # self.decoder = nn.Linear(bert_model_embedding_weights.size(1), i.e embedding size
        # bert_model_embedding_weights.size(0),bias=False) i.e vocab size
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.hidden_size, config.embedding_size
            )

        self.decoder = nn.Linear(
            self.config.embedding_size, self.config.vocab_size, bias=False
        )
        self.decoder.weight = bert_model_embedding_weights
        ## enhancement biad could be added

    def forward(self, hidden_states, pairs):
        bs, num_pairs, x = pairs.size()
        assert x == 2

        bs, seq_len, dim = hidden_states.size()
        # dim is number of hidden states
        assert dim == self.config.hidden_size
        # pair indices: (bs, num_pairs)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(
            hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)
        )
        # pair states: bs * num_pairs, max_span_len, dim
        left_hidden = (
            left_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.max_span_len, 1)
        )
        right_hidden = torch.gather(
            hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim)
        )
        # bs * num_pairs, max_span_len, dimzzz
        right_hidden = (
            right_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.max_span_len, 1)
        )

        assert right_hidden.size() == (bs * num_pairs, self.max_span_len, dim)
        # (max_span_len, dim)

        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(
            torch.cat(
                (
                    left_hidden,
                    right_hidden,
                    position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1),
                ),
                -1,
            )
        )
        ##ad if embedding and hidden states are not same
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)
        ## bs*num_pairs, max_span_len, 2*hidden_size+ positional_embedding_size)
        # target scores : bs * num_pairs, max_span_len, vocab_size
        target_scores = self.decoder(hidden_states)

        return target_scores


class SpanElectraDiscrimnatorPredictionHead(nn.Module):
    """get corrosponding hidden states and posiitonal embedding, pass them through MLP with normalization, then through classififer for token classiifcaion"""

    def __init__(self, config, max_span_len=20, position_embedding_size=200):
        super().__init__()
        self.config = config
        self.max_span_len = max_span_len
        self.position_embeddings = nn.Embedding(max_span_len, position_embedding_size)

        # max_span_len = max word pieces b/w a pair
        self.mlp_layer_norm = MLPWithLayerNorm(
            config, config.hidden_size * 2 + position_embedding_size
        )  ##ad different layer change_needed
        # classifier for token
        self.clf = nn.Linear(
            self.config.hidden_size, 2, bias=False
        )  ##ad change_needed, different decoder for classiffication will be requi
        ## enhancement biad could be added

    def forward(self, hidden_states, pairs):
        bs, num_pairs, x = pairs.size()
        assert x == 2

        bs, seq_len, dim = hidden_states.size()
        # dim is number of hidden states
        # pair indices: (bs, num_pairs)
        assert dim == self.config.hidden_size

        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(
            hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)
        )
        # pair states: bs * num_pairs, max_span_len, dim
        left_hidden = (
            left_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.max_span_len, 1)
        )
        right_hidden = torch.gather(
            hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim)
        )
        # bs * num_pairs, max_span_len, dim
        right_hidden = (
            right_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.max_span_len, 1)
        )

        assert right_hidden.size() == (bs * num_pairs, self.max_span_len, dim)
        # (max_span_len, dim)

        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(
            torch.cat(
                (
                    left_hidden,
                    right_hidden,
                    position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1),
                ),
                -1,
            )
        )  ##ad change_needed
        ##ad input is bs*num_pairs, max_span_len, 2*hidden_size+ positional_embedding_size)
        ##ad target scores i.e output: bs * num_pairs, max_span_len, 2
        target_logits = self.clf(hidden_states)
        return target_logits


# @add_start_docstrings(
#     """
#     TBA
#     SPANELECTRA_START_DOCSTRING,
# )


class SpanElectraLMHead(SpanElectraPretrainedModels):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__(config)
        self.config = config
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.hidden_size, config.embedding_size
            )
        self.decoder = nn.Linear(
            self.config.embedding_size, self.config.vocab_size, bias=False
        )
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.mlp_layer_norm(hidden_states)
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        target_scores = self.decoder(hidden_states)
        return target_scores


class SpanElectraGenerator(nn.Module):
    def __init__(self, config):
        ## config should contain max_pair_targets and position embedding size
        super().__init__(config)
        ## max_pair_targets= max_span_len
        self.config = config
        self.target_layer = self.config.target_layer
        self.config.output_hidden_states = True
        self.pad_token_id = self.config.pad_token_id
        self.use_sbo = self.config.use_SBO
        self.spanElectra = SpanElectraModel(self.config)
        self.lm_head = SpanElectraLMHead(
            config=self.config,
            bert_model_embedding_weights=self.spanElectra.get_input_embeddings().weight,
        )

        if self.use_sbo:
            self.sbo_head = SpanElectraGeneratorPredictionHead(
                self.config,
                bert_model_embedding_weights=self.spanElectra.get_input_embeddings().weight,
                max_span_len=self.config.max_span_len,
                position_embedding_size=self.config.position_embedding_size,
            )  ##ad check this embedding initialization.
        self.init_weights()

    # @add_start_docstrings_to_callable(SPANELECTRA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pairs=None,
        labels=None,
        return_logits=False,
    ):
        ## pairs to pred
        t0 = time.time()
        last_hidden_layer, all_hidden_layers = self.spanElectra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
        )
        t1 = time.time()
        outputs = [torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        mlm_score = self.lm_head(last_hidden_layer)
        if return_logits:
            outputs[1] = mlm_score
        t2 = time.time()

        if labels is not None:
            mlm_loss = ceLoss(
                logits=mlm_score, labels=labels, ignore_idx=self.config.pad_token_id
            )
            outputs[0] = mlm_loss
        t3 = time.time()
        # print("encoder calc {}, lm head {} , lm loss {} ".format(t1-t0, t2-t1, t3-t2))
        if self.use_sbo:
            t4 = time.time()
            sbo_score = self.sbo_head(all_hidden_layers[self.target_layer], pairs)
            t5 = time.time()
            sbo_score = get_pre_from_span_level_logits(
                logits=sbo_score,
                pairs=pairs,
                dummy_id=self.config.dummy_id,
                max_span_len=self.config.max_span_len,
                max_seq_len=self.config.max_seq_len,
            )
            t6 = time.time()
            if return_logits:
                outputs[3] = sbo_score
            if labels is not None:
                sbo_loss = ceLoss(
                    logits=sbo_score, labels=labels, ignore_idx=self.config.pad_token_id
                )
                outputs[2] = sbo_loss
            t7 = time.time()
            # print(" sbo head{} , sbo span {}, sbo_loss {} ".format(t5-t4, t6-t5, t7-t6))
        return outputs


# @add_start_docstrings(
#     """
#     TBA""",
#     SPANELECTRA_START_DOCSTRING,
# )


class SpanElectraDiscrimnator(nn.Module):
    def __init__(self, config):
        ## config should contain max_pair_targets and position embedding size
        super().__init__()
        ## max_pair_targets= max_span_len
        self.config = config
        self.pad_token_id = self.config.pad_token_id
        self.target_layer = self.config.target_layer
        self.config.output_hidden_states = True
        self.ignore_label = self.config.ignore_label  # CR
        self.use_sbo = self.config.use_SBO
        self.spanElectra = SpanElectraModel(self.config)
        self.at_head = SpanElectraAllTokenDiscriminatorHead(config)
        if self.use_sbo:
            self.sbo_head = SpanElectraDiscrimnatorPredictionHead(
                self.config,
                max_span_len=self.config.max_span_len,
                position_embedding_size=self.config.position_embedding_size,
            )
        # self.init_weights()

    def set_input_embedding(self, value):
        self.spanElectra.embeddings.word_embeddings.weight = value

    # @add_start_docstrings_to_callable(SPANELECTRA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pairs=None,
        labels=None,
    ):
        ## pairs to pred
        enco_out = self.spanElectra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
        )

        outputs = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        at_logits = self.at_head(enco_out[0])
        at_loss = ceLoss(
            logits=at_logits, labels=labels, ignore_idx=self.config.ignore_label
        )
        outputs[0] = at_loss
        pred_labels = at_logits

        if self.use_sbo:
            sbo_logits = self.sbo_head(enco_out[1][self.target_layer], pairs)
            sbo_logits = get_pre_from_span_level_logits(
                logits=sbo_logits,
                pairs=pairs,
                dummy_id=self.config.dummy_id,
                max_span_len=self.config.max_span_len,
                max_seq_len=self.config.max_seq_len,
            )
            sbo_loss = ceLoss(
                logits=sbo_logits, labels=labels, ignore_idx=self.config.ignore_label
            )
            outputs[1] = sbo_loss
            pred_labels = (pred_labels + sbo_logits) / 2

        pred_labels = get_pre(pred_labels)
        disc_f1 = get_f1(
            orig=labels, pred=pred_labels, ignore_label=self.config.ignore_label
        )
        outputs[2] = torch.tensor(disc_f1, dtype=torch.float)
        return outputs


# India is | my country and  [pad]*17 | I love India     2 X [[pad]*20]
# India is | out country you [some other 17 tokens] | I love India  2 x [[some other tokens]*20]

# sapan ELelctra:
# span electra + electra    ->>> better
# just replace genratore


class SpanElectraAllTokenDiscriminatorHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size)
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size)
        self.clf = nn.Linear(self.config.hidden_size, 2)

    def forward(self, hidden_states):
        hidden_states = self.mlp_layer_norm(hidden_states)
        # print("hs after pooler",hidden_states.size())
        target_scores = self.clf(hidden_states)
        return target_scores


class SpanaElectraJoint(nn.Module):
    def __init__(self, gen_config, disc_config):
        super().__init__()
        self.gen_config = gen_config
        self.disc_config = disc_config
        self.gen_config.output_hidden_states = True
        self.disc_config.output_hidden_states = True
        self.target_layer = self.gen_config.target_layer
        self.all_token_clf = self.disc_config.all_token_clf
        self.pad_token_id = self.gen_config.pad_token_id
        self.dummy_id = self.gen_config.dummy_id

        self.use_sbgo = self.gen_config.use_SBO
        self.use_sbpo = self.disc_config.use_SBO

        self.ignore_label = self.disc_config.ignore_label  # change ignore_label
        if (
            self.gen_config.vocab_size != self.disc_config.vocab_size
            or self.gen_config.max_position_embeddings
            != self.disc_config.max_position_embeddings
        ):
            raise ValueError(
                "vocab size and max_possition emebdding of generator and discrimnator is not same"
            )

        if self.gen_config.embedding_size == self.disc_config.embedding_size:
            self.same_embed = True
            self.embeddings = SpanElectraEmbeddings(self.disc_config)
            if disc_config.embedding_size != disc_config.hidden_size:
                self.disc_embeddings_project = nn.Linear(
                    disc_config.embedding_size, disc_config.hidden_size
                )
            if gen_config.embedding_size != gen_config.hidden_size:
                self.gen_embeddings_project = nn.Linear(
                    gen_config.embedding_size, gen_config.hidden_size
                )

        else:
            self.same_embed = False
            self.gen_embed = SpanElectraEmbeddings(self.gen_config)
            self.disc_embed = SpanElectraEmbeddings(self.disc_config)
            if disc_config.embedding_size != disc_config.hidden_size:
                self.disc_embeddings_project = nn.Linear(
                    disc_config.embedding_size, disc_config.hidden_size
                )
            if gen_config.embedding_size != gen_config.hidden_size:
                self.gen_embeddings_project = nn.Linear(
                    gen_config.embedding_size, gen_config.hidden_size
                )

        if self.check_same_encoder_config():
            self.same_encoder = True
            self.encoder = BertEncoder(self.disc_config)

        else:
            self.same_encoder = False
            self.gen_encoder = BertEncoder(self.gen_config)
            self.disc_encoder = BertEncoder(self.disc_config)

        self.gen_lm_head = SpanElectraLMHead(
            config=self.gen_config,
            bert_model_embedding_weights=self.get_embeddings(part="generator").weight,
        )
        self.disc_at_head = SpanElectraAllTokenDiscriminatorHead(
            config=self.disc_config
        )

        if self.use_sbgo:
            self.gen_sbo_head = SpanElectraGeneratorPredictionHead(
                config=self.gen_config,
                bert_model_embedding_weights=self.get_embeddings(
                    part="generator"
                ).weight,
                max_span_len=self.gen_config.max_span_len,
                position_embedding_size=self.gen_config.position_embedding_size,
            )

        if self.use_sbpo:
            self.disc_sbo_head = SpanElectraDiscrimnatorPredictionHead(
                config=self.disc_config,
                max_span_len=self.disc_config.max_span_len,
                position_embedding_size=self.disc_config.position_embedding_size,
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pairs=None,
        labels=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        head_mask = self.get_head_mask(self.gen_config, head_mask)

        if self.same_embed:
            gen_hidden_states = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
            if hasattr(self, "gen_embeddings_project"):
                gen_hidden_states = self.gen_embeddings_project(gen_hidden_states)

        else:
            gen_hidden_states = self.gen_embed(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
            if hasattr(self, "gen_embeddings_project"):
                gen_hidden_states = self.gen_embeddings_project(gen_hidden_states)

        if self.same_encoder:
            gen_enco_out = self.encoder(
                gen_hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )

        else:
            gen_enco_out = self.gen_encoder(
                gen_hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )
        # gen_lm_loss,  gen_sbo_loss, gen_accu, disc_at_loss, disc_sb0_loss, disc_f1
        outputs = [
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        ]

        gen_lm_logits = self.gen_lm_head(gen_enco_out[0])
        if labels is not None:
            gen_lm_loss = ceLoss(
                logits=gen_lm_logits,
                labels=labels,
                ignore_idx=self.gen_config.pad_token_id,
            )
            outputs[0] = gen_lm_loss

        pred_tokens = gen_lm_logits

        if self.use_sbgo:
            gen_sbo_logits = self.gen_sbo_head(
                gen_enco_out[1][self.target_layer], pairs
            )  # get logits from gen MLMhead
            gen_sbo_logits = get_pre_from_span_level_logits(
                logits=gen_sbo_logits,
                pairs=pairs,
                dummy_id=self.gen_config.dummy_id,
                max_span_len=self.gen_config.max_span_len,
                max_seq_len=self.gen_config.max_seq_len,
            )
            gen_sbo_loss = ceLoss(
                logits=gen_sbo_logits,
                labels=labels,
                ignore_idx=self.gen_config.pad_token_id,
            )
            outputs[1] = gen_sbo_logits
            pred_tokens = (pred_tokens + gen_sbo_logits) / 2

        pred_tokens = get_pre(pred_tokens.detach())
        assert pred_tokens.size() == input_ids.size()

        clf_inputs, disc_at_labels = get_disc_in_disc_labels(
            input_ids=input_ids,
            pred_tokens=pred_tokens,
            orig_labels=labels,
            pad_token_id=self.gen_config.pad_token_id,
            ignore_label=self.gen_config.ignore_label,
        )

        if self.same_embed:
            disc_hidden_states = self.embeddings(
                input_ids=clf_inputs,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
            if hasattr(self, "disc_embeddings_project"):
                disc_hidden_states = self.disc_embeddings_project(disc_hidden_states)

        else:
            disc_hidden_states = self.disc_embed(
                input_ids=clf_inputs,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
            if hasattr(self, "disc_embeddings_project"):
                disc_hidden_states = self.disc_embeddings_project(disc_hidden_states)

        if self.same_encoder:
            disc_enco_out = self.encoder(
                disc_hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )
        else:
            disc_enco_out = self.disc_encoder(
                disc_hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )

        disc_at_logits = self.disc_at_head(disc_enco_out[0])
        disc_at_loss = ceLoss(
            logits=disc_at_logits,
            labels=disc_at_labels,
            ignore_idx=self.disc_config.ignore_label,
        )
        outputs[3] = disc_at_loss
        pred_labels = disc_at_logits

        if self.use_sbpo:
            disc_sbo_logits = self.disc_sbo_head(
                disc_enco_out[1][self.target_layer], pairs
            )

            disc_sbo_logits = get_pre_from_span_level_logits(
                logits=disc_sbo_logits,
                pairs=pairs,
                dummy_id=self.disc_config.dummy_id,
                max_span_len=self.disc_config.max_span_len,
                max_seq_len=self.disc_config.max_seq_len,
            )

            disc_sbo_loss = ceLoss(
                logits=disc_sbo_logits,
                labels=disc_at_labels,
                ignore_idx=self.disc_config.ignore_label,
            )
            outputs[4] = disc_sbo_loss
            pred_labels = (pred_labels + disc_sbo_logits) / 2

        pred_labels = get_pre(pred_labels)
        disc_f1 = get_f1(
            orig=disc_at_labels,
            pred=pred_labels,
            ignore_label=self.disc_config.ignore_label,
        )
        gen_accu = get_flat_acc(
            orig=labels, pred=pred_tokens, ignore_label=self.gen_config.pad_token_id
        )

        outputs[2] = torch.tensor(gen_accu, dtype=torch.float, device=device)
        outputs[5] = torch.tensor(disc_f1, dtype=torch.float, device=device)

        return outputs

    def get_embeddings(self, part=None):
        if self.same_embed:
            return self.embeddings.word_embeddings
        elif part == "generator":
            return self.gen_embed.word_embeddings

        elif part == "discriminator":
            return self.disc_embed.word_embeddings

        else:
            raise ValueError("wrong choice of embedding part")

    def check_same_encoder_config(self):
        # for encoder to be same, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size
        if self.gen_config.hidden_size != self.disc_config.hidden_size:
            return False
        elif self.gen_config.num_hidden_layers != self.disc_config.num_hidden_layers:
            return False

        elif (
            self.gen_config.num_attention_heads != self.disc_config.num_attention_heads
        ):
            return False

        elif self.gen_config.intermediate_size != self.disc_config.intermediate_size:
            return False

        return True

    def get_extended_attention_mask(self, attention_mask, input_shape, device):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, config, head_mask):
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        num_hidden_layers = config.num_hidden_layers
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
