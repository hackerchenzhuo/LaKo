# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb

import numpy as np
import heapq


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def find_index(self, fact_list, start):
        try:
            return fact_list.index(5, start)
        except ValueError:
            return -1

    # 32,120
    def get_attention_score(self, fact_score_list, start, end, style):
        # # mean / max / 21mean /
        if style == "mean":
            return sum(fact_score_list[start:end]) / (end - start)
        elif style == "max":
            return max(fact_score_list[start:end])
        elif style == "21mean":
            num = max(int((end - start + 1) / 2), 1)
            return sum(heapq.nlargest(num, fact_score_list[start:end])) / num

    def get_psg_score(self, psg_scores, context_ids, batch_size, text_maxlength, attention_score_style):

        score_all = []
        for i in range(batch_size):
            # psg_scores_list = [i.item() for i in psg_scores[i]]
            # psg_list = [i.item() for i in context_ids[i][0]]

            psg_scores_list = psg_scores[i][0].cpu().numpy().tolist()
            psg_list = context_ids[i][0].cpu().numpy().tolist()

            start = psg_list.index(10, 3) + 1

            if psg_list[-1] == 0:
                end = psg_list.index(0)
            else:
                end = text_maxlength
            score = np.array([self.get_attention_score(psg_scores_list, start, end, attention_score_style)])

            score_all.append(score)

        score_all = np.array(score_all)
        # score_all = torch.Tensor(score_all)
        score_all = torch.from_numpy(score_all)

        return score_all

    def get_crossattention_scores(self, opt, context_ids, tokenizer, context_mask):
        assert opt.stream == 2
        """
        inherit and modify from:
        Distilling Knowledge from Reader to Retriever:https://arxiv.org/abs/2012.04584.
        """
        fact_num = opt.n_context
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)

        scores = torch.cat(scores, dim=2)
        if opt.use_last_half_layer_attention == "yes":
            _, scores = torch.chunk(scores, 2, dim=2)

        bsz, n_heads, n_layers, _ = scores.size()
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        if opt.stream == 1:
            pass
        elif opt.stream == 2:
            attention_score_style = opt.attention_score_style
            psg_scores, fact__scores = torch.chunk(scores, 2, dim=3)
            fact__scores = fact__scores.sum(dim=[1, 2])
            fact_all_score = []

            for batch in range(bsz):
                # fact_loc = []
                fact_score = []
                # fact_list = list(i.item() for i in context_ids[batch][1])
                # fact_score_list = list(i.item() for i in fact__scores[batch][0])
                fact_list = context_ids[batch][1].cpu().numpy().tolist()
                fact_score_list = fact__scores[batch][0].cpu().numpy().tolist()

                start = 2
                # pdb.set_trace()
                for i in range(fact_num):
                    end = self.find_index(fact_list, start) + 1
                    if end == 0:
                        break

                    fact_score.append(self.get_attention_score(fact_score_list, start, end, attention_score_style))
                    # TODO: max or part of the token?
                    start = end
                if len(fact_score) < fact_num and fact_list[-1] != 0:
                    end = len(fact_list)
                    if end > start:
                        fact_score.append(self.get_attention_score(fact_score_list, start, end, attention_score_style))
                while len(fact_score) < fact_num:

                    fact_score.append(-5)

                assert len(fact_score) == fact_num

                fact_score = np.array(fact_score)
                fact_all_score.append(fact_score)
            fact_all_score = np.array(fact_all_score)
            fact_all_score = torch.from_numpy(fact_all_score)
        layers_and_heads = n_layers * n_heads
        fact_all_score = fact_all_score / layers_and_heads
        return fact_all_score

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1), ) + outputs[1:]
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
    self,
    input,
    mask=None,
    kv=None,
    position_bias=None,
    past_key_value_state=None,
    head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

    # pdb.set_trace()

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:

        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=256,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=130,
                 question_maxlength=130,
                 projection=True,
                 asymmetric_retri=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls = extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection
        self.asymmetric_retri = asymmetric_retri


class Retriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)

        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            print("using proj....")
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        elif self.config.asymmetric_retri == "yes":
            print("using proj_iq and proj_fact....")
            self.proj_iq = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.proj_fact = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )

            self.norm_iq = nn.LayerNorm(self.config.indexing_dimension)
            self.norm_fact = nn.LayerNorm(self.config.indexing_dimension)

        self.loss_fct = torch.nn.KLDivLoss()
        # pdb.set_trace()

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):

        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)

        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            text_type="q",
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            text_type="f",
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, text_type, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)
        elif self.config.asymmetric_retri == "yes":
            if text_type == "q":
                text_output = self.proj_iq(text_output)
                text_output = self.norm_iq(text_output)
            else:
                text_output = self.proj_fact(text_output)
                text_output = self.norm_fact(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        # gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
