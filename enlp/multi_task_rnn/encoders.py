#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import embedding_ops
from tensorflow.contrib import rnn
import tensorflow as tf


###############################  ENCODERS  #####################################################

def encoder_RNN_embed(cell, encoder_inputs, num_encoder_symbols, word_embedding_size, sequence_length,
                      batch_size, context_win_size=1, scope=None, bidirectional_rnn=False, train_embeddings=False,
                      num_decoder_symbols=None, dtype=tf.float32):
  """

  Embedds input and applies the RNN cell

  :return:
  """
  if bidirectional_rnn:
    encoder_cell_fw = cell
    encoder_cell_bw = cell

    embedding = tf.get_variable("embedding", [num_encoder_symbols, word_embedding_size],
                                            trainable=train_embeddings)

    print(embedding)

    encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input)
                               for encoder_input in encoder_inputs]

    if context_win_size > 1:
      context_encoder_embedded_inputs = [tf.reshape(tensor, (-1, context_win_size * word_embedding_size))
                                         for tensor in encoder_embedded_inputs]
    else:
      context_encoder_embedded_inputs = encoder_embedded_inputs

    if num_decoder_symbols:
      zeros = tf.zeros([batch_size, 2*cell.output_size + num_decoder_symbols], dtype=dtype)
      zeros.set_shape([None, 2*cell.output_size + num_decoder_symbols])
      _inputs = [tf.concat(1, [embedded_input, zeros])
                 for embedded_input in context_encoder_embedded_inputs]
    else:
      _inputs = context_encoder_embedded_inputs

    with tf.variable_scope(scope or "encoder_rnn") as internal_rnn_scope:
      encoder_outputs, encoder_state_fw, encoder_state_bw = rnn.static_bidirectional_rnn(encoder_cell_fw,
                                                                                         encoder_cell_bw,
                                                                                          _inputs,
                                                                                          sequence_length=sequence_length,
                                                                                          dtype=dtype,
                                                                                          scope=internal_rnn_scope)

      all_fw = tf.concat(encoder_state_fw, 1)
      all_bw = tf.concat(encoder_state_bw, 1)
      encoder_state = tf.concat([all_fw, all_bw], 1)

  else:
    encoder_cell_fw = cell

    embedding = tf.get_variable("embedding", [num_encoder_symbols, word_embedding_size],
                                            trainable=train_embeddings)

    print(embedding)

    encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input)
                               for encoder_input in encoder_inputs]

    if context_win_size > 1:
      context_encoder_embedded_inputs = [tf.reshape(tensor, (-1, context_win_size * word_embedding_size))
                                         for tensor in encoder_embedded_inputs]
    else:
      context_encoder_embedded_inputs = encoder_embedded_inputs

    if num_decoder_symbols:
      zeros = tf.zeros([batch_size, cell.output_size + num_decoder_symbols], dtype=dtype)
      zeros.set_shape([None, cell.output_size + num_decoder_symbols])

      _inputs = [tf.concat([embedded_input, zeros], 1)
                 for embedded_input in context_encoder_embedded_inputs]
    else:
      _inputs = context_encoder_embedded_inputs

    with tf.variable_scope(scope or "encoder_rnn") as internal_rnn_scope:
      encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell_fw, _inputs,
                                                      sequence_length=sequence_length, dtype=dtype,
                                                      scope=internal_rnn_scope)

    encoder_state = tf.concat(1, encoder_state)

  return context_encoder_embedded_inputs, encoder_outputs, encoder_state


def encoder_embed(encoder_inputs, num_encoder_symbols, word_embedding_size, context_win_size=1,
                  train_embeddings=False):
  """
  Just embedds inputs and returns the embedded sequence
  :return:
  """
  embedding = tf.get_variable("embedding", [num_encoder_symbols, word_embedding_size],
                                          trainable=train_embeddings)
  encoder_embedded_inputs = [embedding_ops.embedding_lookup(embedding, encoder_input)
                             for encoder_input in encoder_inputs]

  if context_win_size > 1:
    context_encoder_embedded_inputs = [tf.reshape(tensor, (-1, context_win_size * word_embedding_size))
                                       for tensor in encoder_embedded_inputs]
  else:
    context_encoder_embedded_inputs = encoder_embedded_inputs

  return context_encoder_embedded_inputs

