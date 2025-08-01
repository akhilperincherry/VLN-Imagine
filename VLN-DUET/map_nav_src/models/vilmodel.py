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
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

import pdb

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


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
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
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
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

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
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
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
        super().__init__()
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
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
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
            ctx_dim = config.hidden_size
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

class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

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
                
                if len(all_nounphrase_embs_for_imagine) != num_np_tokens_for_imagine:
                    print(f'Debugging instr_id: {obs_instr_ids[batch_idx]}')

                if num_nounphrases_in_subinstr > 0:
                    all_nounphrase_embs_for_imagine_tensor = torch.stack(all_nounphrase_embs_for_imagine, dim=0)
                    mean_nounphrase_embs_for_imagine = torch.mean(all_nounphrase_embs_for_imagine_tensor, dim=0)
                    
                    align_imagine_embeds[batch_idx, imagine_idx] = imagine_proj #update imagine embeds
                    cosine_loss = 1-torch.nn.functional.cosine_similarity(imagine_proj, mean_nounphrase_embs_for_imagine, dim=-1)
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
    # pdb.set_trace()
    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
    
    return loss

class AlignWithContrastiveLossWithNegativeSamples(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.aux_loss_type == 'contrastive-InfoNCE'
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
            tot_valid_noun_phrases_in_batch_idx = sum(1 for sublist in nounphrase_indices_in_instruction for inner_list in sublist if inner_list)
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

                num_nounphrases_in_subinstr = len(curr_sub_instr_nounphrase_indices)
                num_np_tokens_for_imagine = 0
                for noun_phrase_idx in range(num_nounphrases_in_subinstr):
                    curr_np_index = curr_sub_instr_nounphrase_indices[noun_phrase_idx]
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

                    contrastive_loss = compute_contrastive_loss_infonce(imagine_proj, mean_nounphrase_embs_for_imagine, neg_text_embeds, temperature = self.config.infonce_temperature)
                    # print(f'Cosine_loss: {cosine_loss}, Contrastive_loss: {contrastive_loss}')
                    contrastive_loss_all_imagine.append(contrastive_loss)
        
        if len(contrastive_loss_all_imagine) == 0:
            net_loss = 0
        else:
            net_loss = torch.mean(torch.stack(contrastive_loss_all_imagine, dim=0))
        return net_loss, align_imagine_embeds

class AlignWithContrastiveLossReverie(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_proj = MLPProjectionHead(768, 512, config.hidden_size)
        assert config.aux_loss_type == 'cosine'

        self.config = config
    
    def forward(self, align_txt_embeds=None, txt_masks=None, align_imagine_embeds=None, 
            imagine_masks=None, obs_instr_ids=None):
        
        if align_imagine_embeds is not None:
            batch_size = align_imagine_embeds.size(0)
        else:
            batch_size = 1

        device = next(iter(self.parameters())).device

        max_imagine_len_in_batch = align_imagine_embeds.size(1)
        contrastive_loss_all_imagine = []
        imagine_idx = 0 #only one imagination exists in reverie.
        for batch_idx in range(align_imagine_embeds.size(0)):
            imagine_proj = self.image_proj(align_imagine_embeds[batch_idx, imagine_idx])
            assert imagine_masks[batch_idx], 'Imagine embeds is not valid where embedding addition is being applied.'
            all_tok_embs_for_imagine_list = []

            for token_in_np_idx in range(align_txt_embeds[batch_idx].size(0)):
                if txt_masks[batch_idx, token_in_np_idx] == False:
                    continue
                token_embedding = align_txt_embeds[batch_idx, token_in_np_idx]
                all_tok_embs_for_imagine_list.append(token_embedding)

            if len(all_tok_embs_for_imagine_list) > 0:
                all_nounphrase_embs_for_imagine_tensor = torch.stack(all_tok_embs_for_imagine_list, dim=0)
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


class AlignWithContrastiveLossWithNegativeSamplesReverie(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_proj = MLPProjectionHead(768, 512, config.hidden_size)
        assert config.aux_loss_type == 'contrastive-InfoNCE' or config.aux_loss_type == 'constrastive-margin'
        self.config = config
    
    def forward(self, align_txt_embeds=None, txt_masks=None, align_imagine_embeds=None, 
            imagine_masks=None, obs_instr_ids=None):
        
        if align_imagine_embeds is not None:
            batch_size = align_imagine_embeds.size(0)
        else:
            batch_size = 1

        device = next(iter(self.parameters())).device

        all_instr_embeds_dict = {} #store mean text embedding of all instrs in batch to use later while constructing pos and neg embeddings.
        for batch_idx in range(align_imagine_embeds.size(0)):
            all_tok_embs_for_imagine_list = [] #instr tokens for current batch element.

            for token_in_instr_idx in range(align_txt_embeds[batch_idx].size(0)):
                if txt_masks[batch_idx, token_in_instr_idx] == False:
                    continue
                token_embedding = align_txt_embeds[batch_idx, token_in_instr_idx]
                all_tok_embs_for_imagine_list.append(token_embedding)

            if len(all_tok_embs_for_imagine_list) > 0:
                all_tok_embs_for_imagine_tensor = torch.stack(all_tok_embs_for_imagine_list, dim=0)
                mean_text_emb_for_imagine = torch.mean(all_tok_embs_for_imagine_tensor, dim=0)
                all_instr_embeds_dict[batch_idx] = mean_text_emb_for_imagine


        contrastive_loss_all_imagine = []
        imagine_idx = 0 #only one imagination exists in reverie.
        for batch_idx in range(align_imagine_embeds.size(0)):

            neg_text_embeds = [embed for idx_as_key, embed in all_instr_embeds_dict.items() if idx_as_key != batch_idx]
            imagine_proj = self.image_proj(align_imagine_embeds[batch_idx, imagine_idx])
            assert imagine_masks[batch_idx], 'Imagine embeds is not valid where embedding addition is being applied.'
            
            all_tok_embs_for_imagine_list = []
                
            align_imagine_embeds[batch_idx, imagine_idx] = imagine_proj #update imagine embeds
            mean_nounphrase_embs_for_imagine = all_instr_embeds_dict[batch_idx] #positive emb.

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

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
       
    
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)

        if self.config.imagine_enc_pano:
            if self.config.bypass_imag_encoder:
                self.imagine_embeddings = BypassImagineEmbeddings(config)
            if self.config.use_cosine_aux_loss or self.config.no_loss_test:
                if self.config.aux_loss_type == 'cosine':
                    if self.config.dataset == 'reverie':
                        self.contrastive_alignment_model = AlignWithContrastiveLossReverie(config)
                    else:
                        self.contrastive_alignment_model = AlignWithContrastiveLoss(config)
                elif self.config.aux_loss_type == 'contrastive-InfoNCE':
                    if self.config.dataset == 'reverie':
                        self.contrastive_alignment_model = AlignWithContrastiveLossWithNegativeSamplesReverie(config)
                    else:
                        self.contrastive_alignment_model = AlignWithContrastiveLossWithNegativeSamples(config)        
        
        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_imagination(self, imagine_feats, imagine_masks):
        assert imagine_feats is not None
        if self.config.bypass_imag_encoder:
            imagine_embeds = self.imagine_embeddings(imagine_feats)
        return imagine_embeds

    def forward_panorama_per_step(
        self, view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens
    ):
        device = view_img_fts.device
        has_obj = obj_img_fts is not None

        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        if has_obj:
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens

        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks

    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids, imagine_embeds = None, imagine_masks = None
    ):
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)

        if self.config.imagine_enc_pano and self.config.concat_imagine_with == 'language':
            assert imagine_embeds is not None and imagine_masks is not None

            txt_imagine_embeds = torch.cat([txt_embeds, imagine_embeds], 1)
            txt_imagine_masks = torch.cat([txt_masks, imagine_masks], 1)

            # global encoder
            gmap_embeds = self.global_encoder.encoder(
                txt_imagine_embeds, txt_imagine_masks, gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )
            # local encoder
            vp_embeds = self.local_encoder.encoder(txt_imagine_embeds, txt_imagine_masks, vp_embeds, vp_masks)
        else:
            assert self.config.imagine_enc_pano == False
            # global encoder
            gmap_embeds = self.global_encoder.encoder(
                txt_embeds, txt_masks, gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )
            # local encoder
            vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
       
        # local branch
        # vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        # vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
            
        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])

        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            return txt_embeds

        elif mode == 'imagine':
            assert self.config.imagine_enc_pano
            imagine_embeds = self.forward_imagination(batch['imagine_feats'], batch['imagine_masks'])
            return imagine_embeds

        elif mode == 'align_with_contrastive_loss':
            assert self.config.imagine_enc_pano
            if self.config.fix_lang_inside_cosine_model:
                if self.config.dataset == 'reverie':
                    contrastive_loss, aligned_imagine_embeds = self.contrastive_alignment_model(batch['align_txt_embeds'].detach(), batch['txt_masks'], batch['align_imagine_embeds'], 
                    batch['imagine_masks'], batch['obs_instr_ids'])
                else:
                    contrastive_loss, aligned_imagine_embeds = self.contrastive_alignment_model(batch['align_txt_embeds'].detach(), batch['txt_masks'], batch['align_imagine_embeds'], 
                    batch['imagine_masks'], batch['sub_instr_segs'], batch['sub_instr_imag_flag'], batch['noun_phrase_segs'], batch['obs_instr_ids'])
            else:
                if self.config.dataset == 'reverie':
                    contrastive_loss, aligned_imagine_embeds = self.contrastive_alignment_model(batch['align_txt_embeds'], batch['txt_masks'], batch['align_imagine_embeds'], 
                    batch['imagine_masks'], batch['obs_instr_ids'])
                else:
                    contrastive_loss, aligned_imagine_embeds = self.contrastive_alignment_model(batch['align_txt_embeds'], batch['txt_masks'], batch['align_imagine_embeds'], 
                    batch['imagine_masks'], batch['sub_instr_segs'], batch['sub_instr_imag_flag'], batch['noun_phrase_segs'], batch['obs_instr_ids'])
            return contrastive_loss, aligned_imagine_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            if(self.config.imagine_enc_pano):
                return self.forward_navigation_per_step(
                    batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                    batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                    batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                    batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                    batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],imagine_embeds = batch['imagine_embeds'], imagine_masks = batch['imagine_masks']
                )
            else:
                return self.forward_navigation_per_step(
                    batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                    batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                    batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                    batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                    batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids']
                )

            
       