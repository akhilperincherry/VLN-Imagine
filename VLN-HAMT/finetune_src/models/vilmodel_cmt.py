import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from transformers import logging as transformerLogging
transformerLogging.set_verbosity_error()

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

import pdb
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1) #input_ids: 8,250
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) #shape [250]
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #8, 250
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size) #768, 768
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #12, 64
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states) #hidden_states: batch, pano_len, 768 (8, 36, 768); mask: batch, 1, 1, pano_len
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) #8, 12, 36, 64
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #8, 12, 36, 64 and 8, 12, 64, 36 = 8, 12, 36, 36
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask #TODO: 8, 12, 36, 36 + 8, 1, 1, 36 (12 attention heads and each of them generates scores of 36x36 of how much attention each of the 36 image should pay to all the 36 images.)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) #8, 12, 36, 36

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) #8, 12, 36, 64

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #8, 36, 12, 64
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) #8, 36, 768

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add (concat) attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions #bool
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, lang_query_scores = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
            lang_query_scores_probs = nn.Softmax(dim=-1)(lang_query_scores)
        visn_att_output, visual_query_scores = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask) #query, key/val, mask
        visual_query_scores_probs = nn.Softmax(dim=-1)(visual_query_scores)

        if self.no_lang_ca:
            return lang_att_output, visn_att_output
        return lang_att_output, visn_att_output, lang_query_scores_probs, visual_query_scores_probs

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        if self.no_lang_ca:
            lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        else:
            lang_att_output, visn_att_output, lang_query_probs, visual_query_probs = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        # pdb.set_trace()
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask) #outputs are tuples here, not direct tensors.
        #obtain self-attention probs.
        lang_self_attn_probs = nn.Softmax(dim=-1)(lang_att_output[1])
        visual_self_attn_probs = nn.Softmax(dim=-1)(visn_att_output[1])
        
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        if self.no_lang_ca:
            return lang_output, visn_output
        return lang_output, visn_output, lang_query_probs, visual_query_probs, lang_self_attn_probs, visual_self_attn_probs

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)
        
        cross_attention_scores = []
        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds, lang_query_probs, visual_query_probs = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks)
            # cross_attention_scores.append((lang_query_scores, visual_query_scores))

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
        return txt_embeds, hist_embeds, img_embeds #, cross_attention_scores



class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HistoryEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def forward(self, img_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_ang_feats=None):
        '''Args:
        - img_feats: (batch_size, dim_feat)
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, pano_len, dim_feat)
        '''
        device = next(iter(self.parameters())).device
        if img_feats is not None:
            batch_size = img_feats.size(0)
        else:
            batch_size = 1

        type_ids = torch.zeros((batch_size, )).long().to(device)
        type_embeddings = self.type_embedding(type_ids)

        if img_feats is None:
            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
            return cls_embeddings

        # history embedding per step
        embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                     self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                     self.position_embeddings(pos_ids) + \
                     type_embeddings

        if self.pano_encoder is not None:
            # print('pano_img_feats: ', pano_img_feats.shape) #batch, 36, 512
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                              self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
            pano_embeddings = self.dropout(pano_embeddings) #batch, 36, 768
            # TODO: mask is always True
            batch_size, pano_len, _ = pano_img_feats.size()
            extended_pano_masks = torch.zeros(batch_size, pano_len).float().to(device).unsqueeze(1).unsqueeze(2) #batch, 1, 1, 36
            pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0] #batch, 36, 768
            pano_embeddings = torch.mean(pano_embeddings, 1) #8, 768

            embeddings = embeddings + pano_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BypassImagineEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type_embedding = nn.Embedding(1, config.hidden_size)

    def forward(self, imagine_feat): #8, n, 768
        device = next(iter(self.parameters())).device
        batch_size = imagine_feat.size(0)
        type_ids = torch.zeros((batch_size, 1)).long().to(device)
        type_embeddings = self.type_embedding(type_ids) #8, 1, 768
        embeddings = imagine_feat + type_embeddings
        return embeddings

#TODO: pano is a misnomer as this now takes diffusion single view images.
class ImagineEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # print('Max imagine steps steps - ', config.max_imagination_len)
        self.position_embeddings = nn.Embedding(config.max_imagination_len, config.hidden_size)
        # special type embedding for Imaginations
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.imagine_enc_pano = config.imagine_enc_pano
        if config.imagine_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None
            
        self.hidden_size = config.hidden_size
        self.max_imagination_len = config.max_imagination_len

    def forward(self, pano_img_feats, imagine_masks, pos_ids=None):
        '''Args:
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, imagine_len, dim_feat)
        - imagine_masks: (batch_size, imagine_len)
        '''
        device = next(iter(self.parameters())).device
        if pano_img_feats is not None:
            batch_size = pano_img_feats.size(0)
        else:
            batch_size = 1

        imagine_len = pano_img_feats.size(1)

        type_ids = torch.zeros((batch_size, )).long().to(device) #The input to nn.Embeddings is a batch of indices and the corresponding embedding is pulled out. Here there is only one element to pick out.
        type_ids = type_ids.unsqueeze(1).expand(batch_size, imagine_len)       
        type_embeddings = self.type_embedding(type_ids) #batch, imagine_len, 768

        if pos_ids is None:
            pos_ids = torch.arange(imagine_len, dtype=torch.long, device=device) #shape: [imagine_len]
            pos_ids = pos_ids.unsqueeze(0).expand(batch_size, imagine_len) #[batch, imagine_len]
        position_embeddings = self.position_embeddings(pos_ids)  #batch, imagine_len, 768

        pano_img_feats = pano_img_feats + position_embeddings + type_embeddings

        if self.pano_encoder is not None:
            assert(pano_img_feats.size(1) < self.max_imagination_len), "imagination length out of bounds."
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) #batch, 25, 768
            pano_embeddings = self.dropout(pano_embeddings) #batch, 25, 768
            batch_size, imagine_len, _ = pano_img_feats.size()

            assert imagine_masks.shape == (batch_size, imagine_len)
            extended_imagine_masks = imagine_masks.float().to(device).unsqueeze(1).unsqueeze(2) #batch, 1, 1, imagine_len;
            extended_imagine_masks = (1.0 - extended_imagine_masks) * -10000.0

            imagined_pano_embeddings = self.pano_encoder(pano_embeddings, extended_imagine_masks)[0] # This should be batch, imagine_len, 768
            
        else:
            print('imagine_encoder can\'t be none.')
        
        embeddings = imagined_pano_embeddings

        embeddings = self.layer_norm(embeddings) #[8, 14, 768]
        embeddings = self.dropout(embeddings)
        return embeddings

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(self.dropout(x))

class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjectionHead, self).__init__()
        self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)          # Apply dropout
        x = self.relu(self.fc1(x))   # First hidden layer + ReLU
        x = self.relu(self.fc2(x))   # Second hidden layer + ReLU
        x = self.fc3(x)              # Output layer (no activation for final output)
        return x

class AlignWithContrastiveLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_proj = MLPProjectionHead(768, 512, config.hidden_size)
        assert config.aux_loss_type == 'cosine'
        self.config = config
    
    def forward(self, align_txt_embeds=None, txt_masks=None, align_imagine_embeds=None, 
            imagine_masks=None, sub_instr_segs=None, sub_instr_imag_flag=None, noun_phrase_segs=None, obs_instr_ids=None):
        
        if align_imagine_embeds is not None:
            batch_size = align_imagine_embeds.size(0)
        else:
            batch_size = 1

        device = next(iter(self.parameters())).device

        max_imagine_len_in_batch = align_imagine_embeds.size(1)
        contrastive_loss_all_imagine = []

        for batch_idx in range(align_imagine_embeds.size(0)):
            # print(f'Current instr id: {obs_instr_ids[batch_idx]}')
            curr_sub_instr_imag_flag_bool = [x == 'True' for x in sub_instr_imag_flag[batch_idx]]
            assert len(curr_sub_instr_imag_flag_bool) == len(sub_instr_segs[batch_idx])
            assert len(curr_sub_instr_imag_flag_bool) == len(noun_phrase_segs[batch_idx])
            for imagine_idx in range(len(curr_sub_instr_imag_flag_bool)):
                if not curr_sub_instr_imag_flag_bool[imagine_idx]: #if no imaginations.
                    continue
                imagine_proj = self.image_proj(align_imagine_embeds[batch_idx, imagine_idx])
                assert imagine_masks[batch_idx, imagine_idx], 'Imagine embeds is not valid where embedding addition is being applied.'
                all_nounphrase_embs_for_imagine = []
                curr_sub_instr_seg = sub_instr_segs[batch_idx][imagine_idx]
                curr_sub_instr_nounphrase_indices = noun_phrase_segs[batch_idx][imagine_idx]
                # print('Sub-instr indices: ', curr_sub_instr_seg)
                num_nounphrases_in_subinstr = len(curr_sub_instr_nounphrase_indices)
                num_np_tokens_for_imagine = 0
                for noun_phrase_idx in range(num_nounphrases_in_subinstr):
                    curr_np_index = curr_sub_instr_nounphrase_indices[noun_phrase_idx]
                    # print('Current index: ', curr_np_index)
                    # print('Text embeds shape: ', align_txt_embeds.shape)
                    assert curr_np_index[0] >= curr_sub_instr_seg[0] and curr_np_index[1] <= curr_sub_instr_seg[1], 'check np indices and sub-instr indices. They seem off.'
                    num_np_tokens_for_imagine += curr_np_index[1] - curr_np_index[0] + 1
                    for token_in_np_idx in range(curr_np_index[0], curr_np_index[1] + 1):
                        token_embedding = align_txt_embeds[batch_idx, token_in_np_idx]
                        all_nounphrase_embs_for_imagine.append(token_embedding)
                    assert all(txt_masks[batch_idx, curr_np_index[0]:curr_np_index[1]+1]), 'Text_embeds is not valid where embedding addition is being applied.'

                if num_nounphrases_in_subinstr > 0:
                    all_nounphrase_embs_for_imagine_tensor = torch.stack(all_nounphrase_embs_for_imagine, dim=0)
                    mean_nounphrase_embs_for_imagine = torch.mean(all_nounphrase_embs_for_imagine_tensor, dim=0)
                    
                    align_imagine_embeds[batch_idx, imagine_idx] = imagine_proj #update imagine embeds
                    cosine_loss = 1-torch.nn.functional.cosine_similarity(imagine_proj, mean_nounphrase_embs_for_imagine, dim=-1)
                    # print(f'Cosine loss: {cosine_loss}')
                    contrastive_loss_all_imagine.append(cosine_loss)
        if len(contrastive_loss_all_imagine) == 0:
            net_loss = 0
        else:
            net_loss = torch.mean(torch.stack(contrastive_loss_all_imagine, dim=0))

        return net_loss, align_imagine_embeds


def compute_contrastive_loss_infonce(image_embeds, pos_text_embeds, neg_text_embeds_list, temperature=0.3):
    """
    Compute contrastive loss for a single image embedding with a positive and a list of negative text embeddings.
    
    Parameters:
        image_embeds (torch.Tensor): Tensor of shape (768,) representing the image embedding.
        pos_text_embeds (torch.Tensor): Tensor of shape (768,) representing the positive text embedding.
        neg_text_embeds_list (list of torch.Tensor): List of tensors, each of shape (768,), representing negative text embeddings.
        temperature (float): Temperature scaling factor for the contrastive loss.
        
    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    device = image_embeds.device
    if  image_embeds.shape != (1, 768):
        image_embeds = image_embeds.unsqueeze(0).to(device)  # Now (1, 768)
    pos_text_embeds = pos_text_embeds.unsqueeze(0).to(device)  # Now (1, 768)
    
    # Concatenate the positive and negative text embeddings into one tensor
    all_text_embeds = torch.cat([pos_text_embeds] + [neg.unsqueeze(0).to(device) for neg in neg_text_embeds_list], dim=0)  # (1 + num_neg, 768)
    # print(f'Inside contrastive: concatenated embeds shape: {all_text_embeds.shape}')
    
    # Compute cosine similarity
    sim_matrix = torch.nn.functional.cosine_similarity(image_embeds, all_text_embeds) / temperature  # (1 + num_neg,)
    sim_matrix = sim_matrix.unsqueeze(0)  # Now (1, 1 + num_neg)
    
    # Create labels: positive (1) at index 0, negatives (0) otherwise
    labels = torch.zeros(1, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
    
    return loss

def compute_contrastive_loss_margin(image_embeds, pos_text_embeds, neg_text_embeds_list, margin=1.0):
    """
    Compute contrastive loss for a single image embedding with a positive and a list of negative text embeddings.

    Parameters:
        image_embeds (torch.Tensor): Tensor of shape (768,) representing the image embedding.
        pos_text_embeds (torch.Tensor): Tensor of shape (768,) representing the positive text embedding.
        neg_text_embeds_list (list of torch.Tensor): List of tensors, each of shape (768,), representing negative text embeddings.
        margin (float): Margin for negative pairs.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    
    device = image_embeds.device
    
    image_embeds = image_embeds.unsqueeze(0).to(device)  # Now (1, 768)
    pos_text_embeds = pos_text_embeds.unsqueeze(0).to(device)  # Now (1, 768)

    all_text_embeds = torch.cat([pos_text_embeds] + [neg.unsqueeze(0).to(device) for neg in neg_text_embeds_list], dim=0)  # (1 + num_neg, 768)
    
    pos_sim = torch.nn.functional.cosine_similarity(image_embeds, pos_text_embeds).squeeze()  # shape: ()
    neg_sims = torch.nn.functional.cosine_similarity(image_embeds, all_text_embeds[1:])  # shape: (num_neg,)

    pos_loss = 1 - pos_sim

    # Negative loss
    neg_loss = torch.nn.functional.relu(margin + neg_sims - pos_sim)

    loss = pos_loss + torch.mean(neg_loss)  # Average over negative losses

    return loss

class AlignWithContrastiveLossWithNegativeSamples(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.aux_loss_type == 'contrastive-InfoNCE' or config.aux_loss_type == 'constrastive-margin'
        self.image_proj = MLPProjectionHead(768, 512, config.hidden_size)
        self.config = config
    
    def forward(self, align_txt_embeds=None, txt_masks=None, align_imagine_embeds=None, 
            imagine_masks=None, sub_instr_segs=None, sub_instr_imag_flag=None, noun_phrase_segs=None, obs_instr_ids=None):
        
        if align_imagine_embeds is not None:
            batch_size = align_imagine_embeds.size(0)
        else:
            batch_size = 1

        device = next(iter(self.parameters())).device

        #Save all noun-phrase embeds in batch to use during contrastive learning without having to recompute each time in the loop.
        all_noun_phrase_embeds_dict = {}
        for batch_idx in range(align_imagine_embeds.size(0)):
            curr_sub_instr_imag_flag_bool = [x == 'True' for x in sub_instr_imag_flag[batch_idx]]
            all_noun_phrase_embeds_dict[batch_idx] = [] #one text embed for each noun-phrase in batch idx.
            nounphrase_indices_in_instruction = noun_phrase_segs[batch_idx] #outer list corresponds to sub-instrs, inner list is list of noun-phrases in sub-instruction.
            num_subinstrs = len(nounphrase_indices_in_instruction)
            for sub_instr_idx in range(num_subinstrs):
                if not curr_sub_instr_imag_flag_bool[sub_instr_idx]: #use only noun-phrases belonging to sub-instrs that have imaginations for apples to apples.
                    continue
                noun_phrase_indices = nounphrase_indices_in_instruction[sub_instr_idx]
                num_nounphrases= len(noun_phrase_indices)
                
                for noun_phrase_idx in range(num_nounphrases):
                    curr_nounphrase_embs_for_imagine = []
                    curr_np_index = noun_phrase_indices[noun_phrase_idx]
                    for token_in_np_idx in range(curr_np_index[0], curr_np_index[1] + 1):
                        token_embedding = align_txt_embeds[batch_idx, token_in_np_idx]
                        curr_nounphrase_embs_for_imagine.append(token_embedding)
                    if len(curr_nounphrase_embs_for_imagine) > 0:
                        curr_nounphrase_embs_for_imagine_tensor = torch.stack(curr_nounphrase_embs_for_imagine, dim=0)
                        mean_nounphrase_embs_for_imagine = torch.mean(curr_nounphrase_embs_for_imagine_tensor, dim=0)
                        all_noun_phrase_embeds_dict[batch_idx].append(mean_nounphrase_embs_for_imagine)

        max_imagine_len_in_batch = align_imagine_embeds.size(1)
        contrastive_loss_all_imagine = []

        # valid_loss_elements_in_batch = 0
        for batch_idx in range(align_imagine_embeds.size(0)):
            # print(f'Current instr id: {obs_instr_ids[batch_idx]}')
            curr_sub_instr_imag_flag_bool = [x == 'True' for x in sub_instr_imag_flag[batch_idx]]
            neg_text_embeds = [embed for idx_as_key, embeds_as_values in all_noun_phrase_embeds_dict.items() if idx_as_key != batch_idx for embed in embeds_as_values]
            assert len(curr_sub_instr_imag_flag_bool) == len(sub_instr_segs[batch_idx])
            assert len(curr_sub_instr_imag_flag_bool) == len(noun_phrase_segs[batch_idx])
            for imagine_idx in range(len(curr_sub_instr_imag_flag_bool)):
                if not curr_sub_instr_imag_flag_bool[imagine_idx]: #if no imaginations.
                    continue
                imagine_proj = self.image_proj(align_imagine_embeds[batch_idx, imagine_idx])
                assert imagine_masks[batch_idx, imagine_idx], 'Imagine embeds is not valid where embedding addition is being applied.'
                all_nounphrase_embs_for_imagine = []
                curr_sub_instr_seg = sub_instr_segs[batch_idx][imagine_idx]
                curr_sub_instr_nounphrase_indices = noun_phrase_segs[batch_idx][imagine_idx]
                # print('Sub-instr indices: ', curr_sub_instr_seg)
                num_nounphrases_in_subinstr = len(curr_sub_instr_nounphrase_indices)
                num_np_tokens_for_imagine = 0
                for noun_phrase_idx in range(num_nounphrases_in_subinstr):
                    curr_np_index = curr_sub_instr_nounphrase_indices[noun_phrase_idx]
                    # print('Current index: ', curr_np_index)
                    # print('Text embeds shape: ', align_txt_embeds.shape)
                    assert curr_np_index[0] >= curr_sub_instr_seg[0] and curr_np_index[1] <= curr_sub_instr_seg[1], 'check np indices and sub-instr indices. They seem off.'
                    num_np_tokens_for_imagine += curr_np_index[1] - curr_np_index[0] + 1
                    # print(align_txt_embeds[batch_idx, curr_np_index[0]:curr_np_index[1]+1])
                    for token_in_np_idx in range(curr_np_index[0], curr_np_index[1] + 1):
                        token_embedding = align_txt_embeds[batch_idx, token_in_np_idx]
                        all_nounphrase_embs_for_imagine.append(token_embedding)
                    assert all(txt_masks[batch_idx, curr_np_index[0]:curr_np_index[1]+1]), 'Text_embeds is not valid where embedding addition is being applied.'

                if num_nounphrases_in_subinstr > 0:
                    all_nounphrase_embs_for_imagine_tensor = torch.stack(all_nounphrase_embs_for_imagine, dim=0)
                    mean_nounphrase_embs_for_imagine = torch.mean(all_nounphrase_embs_for_imagine_tensor, dim=0)
                    
                    align_imagine_embeds[batch_idx, imagine_idx] = imagine_proj #update imagine embeds

                    if self.config.aux_loss_type == 'contrastive-InfoNCE':
                        contrastive_loss = compute_contrastive_loss_infonce(imagine_proj, mean_nounphrase_embs_for_imagine, neg_text_embeds, temperature = self.config.infonce_temperature)
                    elif self.config.aux_loss_type == 'constrastive-margin':
                        contrastive_loss = compute_contrastive_loss_margin(imagine_proj, mean_nounphrase_embs_for_imagine, neg_text_embeds, margin=self.config.contrastive_margin_value)
                    # print(f'Cosine_loss: {cosine_loss}, Contrastive_loss: {contrastive_loss}')
                    contrastive_loss_all_imagine.append(contrastive_loss)
        if len(contrastive_loss_all_imagine) == 0:
            net_loss = 0
        else:
            net_loss = torch.mean(torch.stack(contrastive_loss_all_imagine, dim=0))

        return net_loss, align_imagine_embeds


class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class NavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.hist_embeddings = HistoryEmbeddings(config)

        if self.config.imagine_enc_pano and (self.config.use_cosine_aux_loss or self.config.no_loss_test):
            if self.config.aux_loss_type == 'cosine':
                self.contrastive_alignment_model = AlignWithContrastiveLoss(config)
            elif self.config.aux_loss_type == 'contrastive-InfoNCE' or self.config.aux_loss_type == 'constrastive-margin':
                self.contrastive_alignment_model = AlignWithContrastiveLossWithNegativeSamples(config)

        if self.config.imagine_enc_pano:
            if not self.config.bypass_imag_encoder:
                self.imagine_embeddings = ImagineEmbeddings(config)
            else:
                self.imagine_embeddings = BypassImagineEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.init_weights()

        self.fix_lang_embedding = self.config.fix_lang_embedding
        self.fix_hist_embedding = self.config.fix_hist_embedding
        self.fix_obs_embedding = self.config.fix_obs_embedding
        if self.config.imagine_enc_pano:
            self.fix_imagine_embeds = self.config.fix_imagine_embeds
        
    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None, imagine_pano_img_feats=None, imagine_masks=None, imagine_embeds=None, 
                align_txt_embeds=None, align_imagine_embeds=None, sub_instr_segs=None, sub_instr_imag_flag=None, noun_phrase_segs=None, obs_instr_ids=None, return_cross_attention_probs=False):
        
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2) #txt_masks: [8, 250], extended_txt_masks:[8, 1, 1, 250]
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = torch.zeros_like(txt_ids) #8, 250
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids) #BertEmbeddings() #8, 60 or 250, 768

            for layer_module in self.encoder.layer: #Self-attention transformer. #BertLayer()
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca: # run self-attn layers for lang.
                all_txt_embeds = [txt_embeds] #Note: This line makes embeddings a list which breaks a lot of things I have implemented. Keep this in mind if you ever want to turn this flag on (shouldn't need to).
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                pano_img_feats=hist_pano_img_feats, pano_ang_feats=hist_pano_ang_feats)
            if self.fix_hist_embedding:
                hist_embeds = hist_embeds.detach()
            return hist_embeds
         
        if mode == 'imagine':
            assert imagine_pano_img_feats is not None
            if self.config.bypass_imag_encoder:
                imagine_embeds = self.imagine_embeddings(imagine_pano_img_feats)
            else:
                imagine_embeds = self.imagine_embeddings(imagine_pano_img_feats, imagine_masks) #imagine_pano_img_feats: [8, 14, 36, 512], imagine_masks: [8, 14], imagine_embeds: [8, 14, 36, 768]
            if self.fix_imagine_embeds:
                imagine_embeds = imagine_embeds.detach()
            return imagine_embeds

        if mode == 'align_with_contrastive_loss':
            contrastive_loss, aligned_imagine_embeds = self.contrastive_alignment_model(align_txt_embeds=align_txt_embeds, txt_masks=txt_masks, align_imagine_embeds=align_imagine_embeds, 
            imagine_masks=imagine_masks, sub_instr_segs=sub_instr_segs, sub_instr_imag_flag=sub_instr_imag_flag, noun_phrase_segs=noun_phrase_segs, obs_instr_ids=obs_instr_ids)
            return contrastive_loss, aligned_imagine_embeds

        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding (this is across time and not per step)
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2) #hist_masks: [8,1] with True, extended_hist_masks: [batch_size, 1, 1, 1]
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            # This is the temporal history transformer
            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers: #num_h layers = 0 though.
                    temp_output = layer_module(hist_embeds, extended_hist_masks) #hist_embeds: [8, 1, 768]; 1 is at t=0 because there is no history yet.
                    hist_embeds = temp_output[0] #batch_size, 1, 768

            # image obs embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2) #ob_masks: [8, 38]; extended_ob_masks: [batch, 1, 1, 38]
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = torch.ones(ob_img_feats.size(0), ob_img_feats.size(1), dtype=torch.long, device=self.device) #batch, 38
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types) #ob_embeds: [batch, 38, 768]
            if self.encoder.r_layers is not None: #r_layers is also 0.
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0] #batch, 38, 768
            if self.fix_obs_embedding:
                ob_embeds = ob_embeds.detach()

            # multi-modal encoding
            hist_max_len = hist_embeds.size(1) #hist_embeds:[batch, time-steps, 768]
            hist_ob_embeds = torch.cat([hist_embeds, ob_embeds], 1) #[batch, time-steps + 38, 768]; ob_embeds are along the nImages in panoramas and hist_embeds is along the time axis.
            extended_hist_ob_masks = torch.cat([extended_hist_masks, extended_ob_masks], -1) #[8, 1, 1, 39]

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2) #txt_masks: [8, 250], extended_txt_masks: [8, 1, 1, 250]
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            if isinstance(txt_embeds, list): #if no lang_ca is used.
                txt_len = txt_embeds[0].size(1) #250
            else:
                txt_len = txt_embeds.size(1)

            if(self.config.imagine_enc_pano):
                assert(imagine_embeds is not None)
                # Preparing imagination embeddings for mult-modal temporal transformer encoding
                extended_imagine_masks = imagine_masks.unsqueeze(1).unsqueeze(2) #imagine_masks: [8, 14]; extended_imagine_masks: [batch, 1, 1, 14].
                extended_imagine_masks = extended_imagine_masks.to(dtype=self.dtype)
                extended_imagine_masks = (1.0 - extended_imagine_masks) * -10000.0
                
                if self.config.concat_imagine_with == 'visual':
                    hist_ob_imagine_embeds = torch.cat([hist_ob_embeds, imagine_embeds], 1) #[8, 1+38+14, 768]
                    extended_hist_ob_imagine_masks = torch.cat([extended_hist_ob_masks, extended_imagine_masks], -1) #[8, 1, 1, 53]
                elif self.config.concat_imagine_with == 'language':
                    txt_imagine_embeds = torch.cat([txt_embeds, imagine_embeds], 1) #[8, 1+38+14, 768]
                    # extended_hist_ob_imagine_masks = torch.cat([extended_hist_ob_masks, extended_imagine_masks], -1) #[8, 1, 1, 53]
                    extended_txt_imagine_masks = torch.cat([extended_txt_masks, extended_imagine_masks], -1) #[8, 1, 1, 250+14]
                
            cross_attn_scores_all_layers = []
            self_attn_scores_all_layers = []
            if(self.config.imagine_enc_pano):
                if self.config.concat_imagine_with == 'visual':
                    if self.config.no_lang_ca:
                        all_txt_embeds = txt_embeds #txt_embeds: list of 5 elements each a tensor of shape [8, 250, 768]. Not sure where the 5 comes from.
                        for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers
                            txt_embeds = all_txt_embeds[l]
                            #With addition of attention probs, this might break no_lang_ca if this flag is needed.                  
                            txt_embeds, hist_ob_imagine_embeds = layer_module( #txt_imagine_embeds: [8, 264, 768], hist_ob_embeds: [8, 39, 768]
                                txt_embeds, extended_txt_masks, 
                                hist_ob_imagine_embeds, extended_hist_ob_imagine_masks,
                            )
                    else:
                        for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers         
                            txt_embeds, hist_ob_imagine_embeds, lang_query_scores, visual_query_scores, lang_self_attn_probs, visual_self_attn_probs = layer_module( #txt_imagine_embeds: [8, 264, 768], hist_ob_embeds: [8, 39, 768]
                                txt_embeds, extended_txt_masks, 
                                hist_ob_imagine_embeds, extended_hist_ob_imagine_masks,
                            )
                            cross_attn_scores_all_layers.append((lang_query_scores, visual_query_scores)) #all heads (12)
                            self_attn_scores_all_layers.append((lang_self_attn_probs, visual_self_attn_probs))

                elif self.config.concat_imagine_with == 'language':
                    if self.config.no_lang_ca:
                        all_txt_embeds = txt_embeds #txt_embeds: list of 5 elements each a tensor of shape [8, 250, 768]. Not sure where the 5 comes from.
                        for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers
                            txt_embeds = all_txt_embeds[l]
                            #With addition of attention probs, this might break no_lang_ca if this flag is needed.                  
                            txt_imagine_embeds, hist_ob_embeds = layer_module( #txt_imagine_embeds: [8, 264, 768], hist_ob_embeds: [8, 39, 768]
                                txt_imagine_embeds, extended_txt_imagine_masks, 
                                hist_ob_embeds, extended_hist_ob_masks,
                            )
                    else:
                        for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers         
                            txt_imagine_embeds, hist_ob_embeds, lang_query_scores, visual_query_scores, lang_self_attn_probs, visual_self_attn_probs = layer_module( #txt_imagine_embeds: [8, 264, 768], hist_ob_embeds: [8, 39, 768]
                                txt_imagine_embeds, extended_txt_imagine_masks, 
                                hist_ob_embeds, extended_hist_ob_masks,
                            )
                            cross_attn_scores_all_layers.append((lang_query_scores, visual_query_scores)) #all heads (12)
                            self_attn_scores_all_layers.append((lang_self_attn_probs, visual_self_attn_probs))


            else:
                if self.config.no_lang_ca:
                    all_txt_embeds = txt_embeds #txt_embeds: list of 5 elements each a tensor of shape [8, 250, 768]. Not sure where the 5 comes from.
                    for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers
                        txt_embeds = all_txt_embeds[l] #Seems like after each iteration, the output txt_embeds are being thrown away? Is this because they don't want cross-attention o/ps?
                        txt_embeds, hist_ob_embeds = layer_module( #txt_embeds: [8, 250, 768]
                            txt_embeds, extended_txt_masks, 
                            hist_ob_embeds, extended_hist_ob_masks,
                        )
                else:
                    for l, layer_module in enumerate(self.encoder.x_layers): #4 x-layers
                        txt_embeds, hist_ob_embeds, _, _, _, _ = layer_module( #txt_embeds: [8, 250, 768]
                            txt_embeds, extended_txt_masks, 
                            hist_ob_embeds, extended_hist_ob_masks,
                        )

            ob_max_len = ob_embeds.size(1) #[8, 38, 768]
            if(self.config.imagine_enc_pano):
                if self.config.concat_imagine_with == 'visual':
                    hist_embeds = hist_ob_imagine_embeds[:, :hist_max_len] #hist_ob_embeds:[8, 1, 768]
                    ob_embeds = hist_ob_imagine_embeds[:, hist_max_len:hist_max_len+ob_max_len] #ob_embeds: [8, 38, 768]
                    imagine_embeds = hist_ob_imagine_embeds[:, hist_max_len+ob_max_len:] #imagine_embeds: [8, 14, 768]
                elif self.config.concat_imagine_with == 'language':
                    hist_embeds = hist_ob_embeds[:, :hist_max_len] #hist_ob_embeds:[8, 1, 768]
                    ob_embeds = hist_ob_embeds[:, hist_max_len:] #ob_embeds: [8, 38, 768]
                    txt_embeds = txt_imagine_embeds[:, :txt_len]
                    imagine_embeds = txt_imagine_embeds[:, txt_len:] #imagine_embeds: [8, 14, 768]
            else:
                hist_embeds = hist_ob_embeds[:, :hist_max_len] #hist_ob_embeds:[8, 1, 768]
                ob_embeds = hist_ob_embeds[:, hist_max_len:] #ob_embeds: [8, 38, 768]
            
            if self.config.no_lang_ca:
                act_logits = self.next_action(ob_embeds).squeeze(-1) # [8, 38, 1] -> [8, 38]
            else:
                if self.config.act_pred_token == 'ob_txt':
                    act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(ob_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1) #[8, 38, 768] * ([8, 1, 768] + [8, 1, 768])
                elif self.config.act_pred_token == 'ob_imagine_text':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + torch.mean(imagine_embeds, 1).unsqueeze(1))).squeeze(-1)
            act_logits.masked_fill_(ob_nav_types==0, -float('inf'))

            if return_cross_attention_probs:
                return act_logits, txt_embeds, hist_embeds, ob_embeds, cross_attn_scores_all_layers, self_attn_scores_all_layers
            else:
                return act_logits, txt_embeds, hist_embeds, ob_embeds

