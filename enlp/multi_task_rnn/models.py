#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from .encoders import encoder_RNN_embed
from .decoders import (seq_labeling_decoder_linear_attention, seq_labeling_decoder_RNN_attention, \
                       seq_classification_decoder_linear)


def dep_attention_RNN(encoder_inputs,
                      encoder_extra_inputs,
                      decoder_inputs,
                      cell,
                      num_encoder_symbols,
                      num_decoder_symbols,
                      word_embedding_size,
                      batch_size,
                      task,
                      sequence_length=None,
                      loop_function=None,
                      bidirectional_rnn=False,
                      context_win_size=1,
                      train_embeddings=True,
                      dtype=dtypes.float32):
  """
  Encoder and decoder share weights.
  """

  assert len(encoder_inputs) == len(decoder_inputs)
  assert len(encoder_inputs) == len(encoder_extra_inputs)

  scope = "dep_attention_RNN"

  enc_outputs = encoder_RNN_embed(cell, encoder_inputs, num_encoder_symbols, word_embedding_size,
                                  sequence_length, batch_size, context_win_size=context_win_size,
                                  scope=scope, train_embeddings=train_embeddings, dtype=dtype,
                                  num_decoder_symbols=num_decoder_symbols, bidirectional_rnn=bidirectional_rnn)

  encoder_embedded_inputs, encoder_outputs, encoder_state = enc_outputs

  if bidirectional_rnn:
    scope += "/FW"

  dec_outputs = seq_labeling_decoder_RNN_attention(cell, num_decoder_symbols, decoder_inputs,
                                                   encoder_state, encoder_outputs,
                                                   batch_size, loop_function, dtype=dtype,
                                                   encoder_embedded_inputs=encoder_embedded_inputs,
                                                   scope=scope, scope_reuse=True)

  decoder_outputs, sequence_attention_weights = dec_outputs

  return decoder_outputs, None


def dep_attention_RNN_linear(encoder_inputs,
                             encoder_extra_inputs,
                             decoder_inputs,
                             cell,
                             num_encoder_symbols,
                             num_labeling_decoder_symbols,
                             num_classification_decoder_symbols,
                             word_embedding_size,
                             batch_size,
                             task,
                             binary_feat_dim=0,
                             sequence_length=None,
                             loop_function=None,
                             bidirectional_rnn=False,
                             context_win_size=1,
                             train_embeddings=True,
                             dtype=dtypes.float32):
  """
  Encoder and decoder share weights.
  """
  enc_outputs = encoder_RNN_embed(cell, encoder_inputs, num_encoder_symbols, word_embedding_size,
                                  sequence_length, batch_size, context_win_size=context_win_size,
                                  train_embeddings=train_embeddings, dtype=dtype,
                                  bidirectional_rnn=bidirectional_rnn)
  # num_decoder_symbols=num_labeling_decoder_symbols,

  encoder_embedded_inputs, encoder_outputs, encoder_state = enc_outputs

  if binary_feat_dim:
    #print(encoder_inputs[0].get_shape())
    #print(encoder_outputs[0].get_shape())
    # set shape for extra inputs
    [tnsr.set_shape((batch_size, binary_feat_dim)) for tnsr in encoder_extra_inputs]
    #print(encoder_extra_inputs[0].get_shape())
    #print(type(encoder_extra_inputs[0]))
    extended_encoder_outputs = []
    for i in xrange(len(encoder_outputs)):
      extended_encoder_outputs.append(array_ops.concat(1, [encoder_outputs[i],
                                                           encoder_extra_inputs[i]]))
  else:
    extended_encoder_outputs = encoder_outputs

  if task["tagging"]:
    tagging_dec_outputs = seq_labeling_decoder_linear_attention(num_labeling_decoder_symbols, extended_encoder_outputs,
                                                                decoder_inputs, encoder_state,
                                                                batch_size, loop_function=loop_function,
                                                                dtype=dtype)
    tagging_decoder_outputs, tagging_attention_weights = tagging_dec_outputs
  else:
    tagging_decoder_outputs, tagging_attention_weights = None, None

  if task['intent']:
    intent_dec_outputs = seq_classification_decoder_linear(num_classification_decoder_symbols,
                                                           extended_encoder_outputs, decoder_inputs,
                                                           encoder_state, batch_size)

    intent_decoder_outputs, intent_attention_weights =  intent_dec_outputs
  else:
    intent_decoder_outputs, intent_attention_weights = None, None

  return tagging_decoder_outputs, tagging_attention_weights, \
         intent_decoder_outputs, intent_attention_weights
