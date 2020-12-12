# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
    * positional encodings are passed in MHattention
"""
import copy
from typing import Optional, List, Union, Callable

import tensorflow as tf
from tf.keras import layers

class FeedForward:
    # Implementation of Feedforward unit
    def __init__(d_model: int, d_ffn: int, activation: Union[str, Callable[..., layer.Layer]] = 'relu',
             ffn_dropout: int = 0.1, use_bias: bool = True, use_glu: bool = False):
        self.d_model = d_model
        self.d_ffd = d_ffn
        self.act = activation
        self.dropout = ffn_dropout
        self.use_bias = use_bias
        self.use_glu = use_glu
        self.l1 = layers.Dense(self.d_ffd, activation=self.act, use_bias=self.use_bias)
        self.d1 = layers.Dropout(self.dropout)
        self.l2 = layers.Dense(self.d_model, activation=None, use_bias=self.use_bias)
        self.d2 = layers.Dropout(self.dropout)
        if self.use_glu:
            self.l3 = layers.Dense(self.d_ffd, activation=None, use_bias=self.use_bias)
            self.d3 = layers.Dropout(self.dropout)

    def __call__(self, x, training):
        act1 = self.d1(self.l1(x), training)
        if self.use_glu:
            act3 = self.d3(self.l3(x), training)
            return self.d2(self.l2(act1 * act3), training)
        return self.d2(self.l2(act1), training)


class ResidWrapper:
    # Implementation of Residual unit
    def __init__(self, fn_layer: layer.Layer, norm_layer: layer.Layer,
             use_prenorm: bool = False, gate_residuals: bool = False):
        self.use_prenorm = use_prenorm
        self.gate_residuals = gate_residuals
        self.fn_layer = fn_layer
        self.norm_layer = norm_layer

    def __call__(self, x, *args, **kwargs):
        # TODO: Implement scalar-gated residuals (Used in Admin initialization and ReZero)
        if self.use_prenorm:
            y = x + self.fn_layer(self.norm_layer(x), *args, **kwargs)
        else:
            y = self.norm_layer(x + self.fn_layer(x, *args, **kwargs))
        return y


class TransformerEncoderDecoder:

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, d_ffn: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[..., layer.Layer]] = 'relu',
                 normalize_before: bool = False, return_intermediate_dec: bool = False):

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        if num_encoder_layers > 0:
            self.encoder_layers = [TransformerEncoderLayer(d_model, nhead, d_ffn, dropout,
                                   ffn_args={'activation': activation},
                                   resid_args={'use_prenorm': normalize_before}) \
                                    for i in range(num_encoder_layers)]
            self.encoder_norm = layers.LayerNormalization() if normalize_before else lambda x: x

        if num_decoder_layers > 0:
            self.decoder_layers = [TransformerDecoderLayer(d_model, nhead, d_ffn, dropout,
                                   ffn_args={'activation': activation},
                                   resid_args={'use_prenorm': normalize_before}) \
                                    for i in range(num_decoder_layers)]
            self.decoder_norm = layers.LayerNormalization()

    def __call__(self, src, mask, query_embed, pos_embed):
        # flatten NxHxWxC to NxHWxC
        src = tf.reshape(src, [src.shape[0], -1, src.shape[-1]])
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = tf.reshape(mask, [mask.shape[0], -1])

        tgt = tf.zeros_like(query_embed)
        memory = self.encoder(src, training=training,
                src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, training=training,
                memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs, memory

    def encoder(self, src: tf.Tensor, training: bool,
                src_mask: Optional[tf.Tensor] = None,
                src_key_padding_mask: Optional[tf.Tensor] = None,
                pos: Optional[tf.Tensor] = None):

        if self.num_encoder_layers <= 0:
            return src

        output = src
        for layer in self.encoder_layers:
            output = layer(output, training=training,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        pos=pos)

        return self.encoder_norm(output)

    def decoder(self, tgt: tf.Tensor, memory: tf.Tensor, training: bool,
                tgt_mask: Optional[tf.Tensor] = None,
                memory_mask: Optional[tf.Tensor] = None,
                tgt_key_padding_mask: Optional[tf.Tensor] = None,
                memory_key_padding_mask: Optional[tf.Tensor] = None,
                pos: Optional[tf.Tensor] = None,
                query_pos: Optional[tf.Tensor] = None):

        if self.num_decoder_layers <= 0:
            return tgt

        output, intermediate = tgt, []
        for layer in self.decoder_layers:
            output = layer(output, memory, training=training,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate_dec:
                intermediate.append(self.decoder_norm(output))

        if self.return_intermediate_dec:
            return tf.stack(intermediate, axis=1)

        return self.decoder_norm(output)


class TransformerBaseLayer:

    def __init__(self, d_model: int, nhead: int, d_ffn: int = 2048, dropout: float = 0.1,
                 attn_layer=layers.Attention: Callable[..., layer.Layer],
                 norm_layer=layers.LayerNormalization: Callable[..., layer.Layer],
                 attn_args: dict = {}, norm_args: dict = {}, resid_args: dict = {},
                 ffn_args: dict = {}, ffn_before: bool = False, ffn_after: bool = True
                 **kwargs):
        # Implementation of Feedforward model
        if ffn_before:
            self.ffn_before = FeedForward(d_model, d_ffn, ffn_dropout=dropout, **ffn_args)
            self.norm_ffn_before = norm_layer(**norm_args)
            self.forward_ffn_before = ResidWrapper(self.ffn_before, self.norm_ffn_before, **resid_args)
        if ffn_after:
            self.ffn_after = FeedForward(d_model, d_ffn, ffn_dropout=dropout, **ffn_args)
            self.norm_ffn_after = norm_layer(**norm_args)
            self.forward_ffn_after = ResidWrapper(self.ffn_after, self.norm_ffn_after, **resid_args)

        self.self_attn = attn_layer(nhead, dropout=dropout, **attn_args)
        self.norm_self_attn = norm_layer(**norm_args)
        self.forward_self_attn = ResidWrapper(self.attn_with_pos, self.norm_self_attn, **resid_args)

    def attn_with_pos(self, x: tf.Tensor, training: bool
                     pos: Optional[tf.Tensor] = None,
                     attn_mask: Optional[tf.Tensor] = None,
                     key_padding_mask: Optional[tf.Tensor] = None):
        q = k = self.with_pos_embed(x, pos)
        return self.self_attn(q, k, value=x, training=training,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask)

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos


class TransformerEncoderLayer(TransformerBaseLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, src, training: bool,
                    src_mask: Optional[tf.Tensor] = None,
                    src_key_padding_mask: Optional[tf.Tensor] = None,
                    pos: Optional[tf.Tensor] = None):
        if self.ffn_before:
            src = self.forward_ffn_before(src, training=training)

        src = self.forward_self_attn(src, training=training,
                     attn_mask=src_mask,
                     key_padding_mask=src_key_padding_mask,
                     pos=pos)

        if self.ffn_after:
            src = self.forward_ffn_after(src, training=training)
        return src


class TransformerDecoderLayer(TransformerBaseLayer):

    def __init__(self, d_model: int, nhead: int, d_ffn: int = 2048, dropout: float = 0.1,
                 attn_layer=layers.Attention: Callable[..., layer.Layer],
                 norm_layer=layers.LayerNormalization: Callable[..., layer.Layer],
                 attn_args: dict = {}, norm_args: dict = {}, resid_args: dict = {},
                 ffn_args: dict = {}, ffn_before: bool = False, ffn_after: bool = True
                 **kwargs):
        super().__init__(d_model, nhead, d_ffn, dropout,
                 attn_layer, norm_layer,
                 attn_args, norm_args, resid_args, ffn_args,
                 ffn_before, ffn_after, **kwargs)
        self.cross_attn = attn_layer(nhead, dropout=dropout, **attn_args)    
        self.norm_cross_attn = norm_layer(**norm_args)
        self.forward_cross_attn = ResidWrapper(self.cross_attn_with_pos, self.norm_cross_attn, **resid_args)

    def cross_attn_with_pos(self, tgt: tf.Tensor, memory: tf.Tensor, training: bool,
                     pos: Optional[tf.Tensor] = None,
                     query_pos: Optional[tf.Tensor] = None,
                     attn_mask: Optional[tf.Tensor] = None,
                     key_padding_mask: Optional[tf.Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        return self.cross_attn(q, k, value=memory, training=training,
                     attn_mask=attn_mask,
                     key_padding_mask=key_padding_mask)

    def __call__(self, tgt: tf.Tensor, memory: tf.Tensor, t: bool,
                tgt_mask: Optional[tf.Tensor] = None,
                memory_mask: Optional[tf.Tensor] = None,
                tgt_key_padding_mask: Optional[tf.Tensor] = None,
                memory_key_padding_mask: Optional[tf.Tensor] = None,
                pos: Optional[tf.Tensor] = None,
                query_pos: Optional[tf.Tensor] = None):
        if self.ffn_before:
            tgt = self.forward_ffn_before(tgt, training=t)

        tgt = self.forward_self_attn(tgt, training=t,
                     attn_mask=tgt_mask,
                     key_padding_mask=tgt_key_padding_mask,
                     pos=query_pos)

        tgt = self.forward_cross_attn(tgt, memory, training=t,
                     memory_mask=memory_mask,
                     memory_key_padding_mask=memory_key_padding_mask,
                     pos=pos, query_pos=query_pos)

        if self.ffn_after:
            tgt = self.forward_ffn_after(tgt, training=t)
        return tgt


def build_transformer(args):
    return TransformerEncoderDecoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        d_ffn=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        use_prenorm=args.pre_norm,
        return_intermediate_dec=True,
    )
