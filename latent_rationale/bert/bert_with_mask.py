from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import BertConfig, is_wandb_available
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler

from .loss import RationaleLoss
from ..common.util import get_z_stats
from ..nn.kuma import RV
from ..nn.kuma_gate import KumaGate

if is_wandb_available():
    import wandb


@dataclass
class BaseModelOutputWithPoolingCrossAttentionsAndZDist(BaseModelOutputWithPoolingAndCrossAttentions):
    z: torch.Tensor = None
    z_dist: RV = None


@dataclass
class RationaleSequenceClassifierOutput(SequenceClassifierOutput):
    z: torch.Tensor = None
    precision: float = None
    recall: float = None


class BertModelWithRationale(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # TODO: Added by @mikimn
        self.z_layer = KumaGate(config.hidden_size)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.rationale_type = config.rationale_type
        assert self.rationale_type in {'all', 'premise', 'hypothesis', 'none', 'supervised'}
        config_dict = config.to_dict()
        if 'rationale_strategy' in config_dict:
            self.rationale_strategy = config.rationale_strategy
        else:
            self.rationale_strategy = 'independent'
        assert self.rationale_strategy in {'independent', 'contextual'}

        self.init_weights()

    def _forward_z_layer_independent(self, z_dist, embeddings, mask=None):
        h = embeddings
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]), h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask.byte(), z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            rationale_mask=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # [B, S, H]
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # TODO Added by @mikimn
        # mask = attention_mask
        fill_mask = None
        if rationale_mask is None:
            if self.rationale_type == 'all':
                rationale_mask = attention_mask  # By default, allow all tokens to be masked
                fill_mask = torch.zeros_like(rationale_mask)
            elif self.rationale_type == 'premise':
                rationale_mask = attention_mask - token_type_ids  # Mask premise
                fill_mask = token_type_ids  # Preserve hypothesis
            elif self.rationale_type == 'hypothesis':
                rationale_mask = token_type_ids  # Mask hypothesis
                fill_mask = attention_mask - token_type_ids  # Preserve premise

        z_dist = None
        z = None
        if rationale_mask is not None and rationale_mask.sum() != 0 and \
                self.rationale_strategy == 'independent':
            z_dist = self.z_layer(embedding_output)
            z = self._forward_z_layer_independent(z_dist, embedding_output, rationale_mask)

            # Ignore padding
            z_mask = (attention_mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            # z_mask = z_mask.clamp(min=0.0, max=1.0)
            # TODO Split attention mask and rationale mask
            # [B, T, H] x [B, T, 1] -> [B, T, H]
            embedding_output = embedding_output * z_mask
            z_mask = z_mask.squeeze(-1)
        else:
            z_mask = torch.ones_like(attention_mask)

        # TODO Remove
        # [B, T, H] x [B, T, 1] => [B, T, H]
        # Hypothesis-only embedding
        # embedding_output = embedding_output * token_type_ids.unsqueeze(-1)

        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [B, T, H]
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] + (z, z_dist)

        return BaseModelOutputWithPoolingCrossAttentionsAndZDist(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            z=z_mask,
            z_dist=z_dist
        )


def extract_rationale(sentence, flag_token='*'):
    if isinstance(sentence, list):
        return [extract_rationale(s, flag_token) for s in sentence]
    candidate = ' '.join([w[1:-1] for w in sentence.split() if w[0] == flag_token and w[-1] == flag_token])
    if len(candidate) == 0:
        return sentence
    return candidate


def rationale_precision_or_recall(input_ids, z, h1, h2, flag_id, flag_token='*', precision=True):
    if flag_id is None:
        return -1
    # h1: p1 <*> p2 <*> ... pt
    # h2: <*> h1 <*> h2 ... pm
    # z:
    # [CLS] p1 p2 ... pt [SEP] h1 h2 ... pm [SEP]
    prec_list = []
    if h1 is None or h2 is None:
        return -1
    for inp, z_pred, premise_highlight, hypothesis_highlight in zip(input_ids, z, h1, h2):
        z_pred = z_pred.cpu().detach().numpy()
        z_ref = [0]  # Start [CLS] is assumed unmasked
        inside_highlight = False
        for i, prem_token in enumerate(premise_highlight):
            if prem_token == flag_id:
                inside_highlight = not inside_highlight
                # start_index = -1 if start_index >= 0 else i
                continue
            # 1 while masking, 0 otherwise
            z_ref.append(int(inside_highlight))
        z_ref.append(0)  # [SEP] assumed unmasked
        inside_highlight = False
        for i, hyp_token in enumerate(hypothesis_highlight):
            if hyp_token == flag_id:
                inside_highlight = not inside_highlight
                # start_index = -1 if start_index >= 0 else i
                continue
            # 1 while masking, 0 otherwise
            z_ref.append(int(inside_highlight))
        z_ref.append(0)  # [SEP] assumed unmasked
        num_pads = (inp == 0).sum()  # Count [PAD] tokens
        z_ref += [0] * num_pads
        if len(z_ref) != len(z_pred):
            # FIXME Ignores samples for which the alignment fails
            continue
        dot_prod = np.array(z_ref).dot(z_pred)
        normalizer = z_pred.sum() if precision else sum(z_ref)
        if normalizer == 0:
            prec_list.append(1.)
        else:
            prec_list.append(dot_prod / normalizer)

    return np.average(prec_list)


def rationale_precision(input_ids, z, h1, h2, flag_id, flag_token='*'):
    return rationale_precision_or_recall(input_ids, z, h1, h2, flag_id, flag_token, precision=True)


def rationale_recall(input_ids, z, h1, h2, flag_id, flag_token='*'):
    return rationale_precision_or_recall(input_ids, z, h1, h2, flag_id, flag_token, precision=False)


class BertWithRationaleForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.config.keys_to_ignore_at_inference = ['precision', 'recall']

        self.bert = BertModelWithRationale(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rationale_loss = RationaleLoss(selection=1.,
                                            lambda_sparsity=config.lambda_init,
                                            lambda_lasso=config.lambda_lasso)
        self.mask_metrics = None
        self.mask_metrics_count = 0

        config_dict = config.to_dict()
        if 'highlight_token' in config_dict:
            self.highlight_token = config.highlight_token
        else:
            self.highlight_token = 1008  # Quickfix (1008 == '*' for BertTokenizer)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            rationale_mask=None,
            premise_highlight=None,
            hypothesis_highlight=None,
            aggregate_mask_metrics=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rationale_mask=rationale_mask,
        )
        if return_dict:
            z, z_dist = outputs.z, outputs.z_dist
        else:
            z, z_dist = outputs[:-2]

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if 'problem_type' not in self.config.__dict__ or self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # loss_fct = MSELoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels)
                raise ValueError("Regression not supported")
            elif self.config.problem_type == "single_label_classification":
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_fct = self.rationale_loss
                z_dists = self.bert.z_dists if z_dist is not None else None
                loss, optional = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), z_dists, attention_mask)
                # z statistics
                if z is not None:
                    num_0, num_c, num_1, total = get_z_stats(z, attention_mask)
                    optional["p0"] = num_0 / float(total)
                    optional["pc"] = num_c / float(total)
                    optional["p1"] = num_1 / float(total)
                    optional["selected"] = optional["pc"] + optional["p1"]
                    if aggregate_mask_metrics:
                        if self.mask_metrics is None:
                            self.mask_metrics = optional
                            self.mask_metrics_count = 1
                        else:
                            self.mask_metrics = {
                                k: v + self.mask_metrics[k] for k, v in optional.items()
                            }
                            self.mask_metrics_count += 1
                # print(optional)
            elif self.config.problem_type == "multi_label_classification":
                # loss_fct = BCEWithLogitsLoss()
                # loss = loss_fct(logits, labels)
                raise ValueError("Multi-label classification not supported")
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not self.training:
            precision = rationale_precision(input_ids, z, hypothesis_highlight, premise_highlight, self.highlight_token)
            recall = rationale_recall(input_ids, z, hypothesis_highlight, premise_highlight, self.highlight_token)
        else:
            precision = None
            recall = None

        return RationaleSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            z=z,
            precision=precision,
            recall=recall
        )
