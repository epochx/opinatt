#!/usr/bin/pythtensorflow.python.ops.rnn_cell on
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attentions import  get_vinyals_attention_function
from .utils import zero_state, _step, _linear
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape
import tensorflow as tf

def seq_labeling_decoder_RNN_attention(cell, num_decoder_symbols, decoder_inputs, att_initial_state,
                                       encoder_outputs, batch_size, loop_function, dtype=tf.float32,
                                       encoder_embedded_inputs=None, scope=None, scope_reuse=False):
  """
  Attention-augmented RNN decoder

   s_i = f(x_i, s_{i-1}, y_{i-1}, c_i)
   o_i = s_i * O
  :return:
  """
  output_size = encoder_outputs[0].get_shape()[1].value

  attention = get_vinyals_attention_function(encoder_outputs, output_size, 1)

  # loop through the encoder_outputs
  attention_encoder_outputs = list()
  sequence_attention_weights = list()

  # rnn initial state for second pass
  state = cell.zero_state(batch_size=batch_size, dtype=dtype)
  for i in xrange(len(encoder_outputs)):
    if i == 0:
      with tf.variable_scope("Initial_Decoder_Attention"):
        proc_att_initial_state = _linear(att_initial_state, output_size, True)
      attn_weights, ds = attention(proc_att_initial_state)
      prev_label = zero_state(num_decoder_symbols, batch_size, dtype=dtype)
    else:
      tf.get_variable_scope().reuse_variables()
      attn_weights, ds = attention(encoder_outputs[i])
      if loop_function:
        with tf.variable_scope("loop_function", reuse=True):
          prev_label = loop_function(prev, i)
      else:
        prev_label = tf.one_hot(decoder_inputs[i - 1], num_decoder_symbols, dtype=dtype)

    if encoder_embedded_inputs:
      # NOTE: here we temporarily assume num_head = 1
      _inputs = tf.concat([encoder_embedded_inputs[i], ds[0], prev_label], 1)
    else:
      _inputs = tf.concat([encoder_outputs[i], ds[0], prev_label], 1)

    with tf.variable_scope(scope or "decoder_rnn", reuse=scope_reuse):
      output, state = cell(_inputs, state)
    # projecting output
    logit = _linear(output, num_decoder_symbols, True)
    if loop_function:
      prev = logit
    attention_encoder_outputs.append(logit)
    sequence_attention_weights.append(attn_weights[0])  # NOTE: here we temporarily assume num_head = 1

  return attention_encoder_outputs, sequence_attention_weights


def seq_labeling_decoder_RNN(cell, decoder_inputs, num_decoder_symbols, batch_size,
                             encoder_inputs=None, loop_function=None, scope=None,
                             dtype=tf.float32):
  """
  Basic RNN decoder

   s_i = f(x_i, s_{i-1}, y_{i-1})
   o_i = s_i * O
  :return:
  """
  output_size = decoder_inputs[0].get_shape()[1].value
  decoder_outputs = []

  # rnn initial state for second pass
  state = cell.zero_state(batch_size=batch_size, dtype=dtype)

  for i in xrange(len(decoder_inputs)):
    if i == 0:
      prev_label = zero_state(num_decoder_symbols, batch_size, dtype=dtype)
    else:
      tf.get_variable_scope().reuse_variables()
      if loop_function:
        with tf.variable_scope("loop_function", reuse=True):
          prev_label = loop_function(prev, i)
      else:
        prev_label = tf.one_hot(decoder_inputs[i - 1], num_decoder_symbols, dtype=dtype)

    if encoder_inputs:
      # NOTE: here we temporarily assume num_head = 1
      _inputs = tf.concat([decoder_inputs[i], encoder_inputs[i], prev_label], 1)
    else:
      _inputs = tf.concat([decoder_inputs[i], prev_label], 1)
    with tf.variable_scope("1st_pass_rnn/FW", reuse=True):
      output, state = cell(_inputs, state)
    # projecting output
    logit = _linear(output, num_decoder_symbols, True)
    if loop_function:
      prev = logit
    decoder_outputs.append(logit)

  return decoder_outputs


def seq_labeling_decoder_linear_attention(num_decoder_symbols, encoder_outputs, decoder_inputs,
                                          att_initial_state, batch_size, loop_function,
                                          dtype=tf.float32, scope=None):


  with tf.variable_scope("seq_labeling_decoder_linear_attention") as scope:

    output_size = encoder_outputs[0].get_shape()[1].value
    attention = get_vinyals_attention_function(encoder_outputs, output_size, 1, scope)

    # loop through the encoder_outputs
    decoder_outputs = list()
    sequence_attention_weights = list()

    # rnn initial state for second pass
    for i in xrange(len(encoder_outputs)):
      if i == 0:
        with tf.variable_scope("Initial_Decoder_Attention"):
          proc_att_initial_state = _linear(att_initial_state, output_size, True)
        attn_weights, ds = attention(proc_att_initial_state)
        prev_label = zero_state(num_decoder_symbols, batch_size, dtype=dtype)
      else:
        tf.get_variable_scope().reuse_variables()
        attn_weights, ds = attention(encoder_outputs[i])
        if loop_function:
          with tf.variable_scope("loop_function", reuse=True):
            prev_label = loop_function(prev, i)
        else:
          prev_label = tf.one_hot(decoder_inputs[i - 1], num_decoder_symbols, dtype=dtype)
      # NOTE: here we temporarily assume num_head = 1
      _inputs = tf.concat([encoder_outputs[i], ds[0], prev_label], 1)
      with tf.variable_scope("AttentionRNNLinearOutput"):
        logit = _linear(_inputs, num_decoder_symbols, True)
      if loop_function:
          prev = logit
      decoder_outputs.append(logit)
      sequence_attention_weights.append(attn_weights[0])  # NOTE: here we temporarily assume num_head = 1

    return decoder_outputs, sequence_attention_weights


def seq_labeling_decoder_linear(decoder_inputs, num_decoder_symbols,
                                scope=None, sequence_length=None, dtype=tf.float32):
  with tf.variable_scope(scope or "non-attention_RNN"):

    decoder_outputs = list()

    # copy over logits once out of sequence_length
    if decoder_inputs[0].get_shape().ndims != 1:
      (fixed_batch_size, output_size) = decoder_inputs[0].get_shape().with_rank(2)
    else:
      fixed_batch_size = decoder_inputs[0].get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = tf.shape(decoder_inputs[0])[0]
    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length is not None:  # Prepare variables
      zero_logit = tf.zeros(
        tf.stack([batch_size, num_decoder_symbols]), decoder_inputs[0].dtype)
      zero_logit.set_shape(
        tensor_shape.TensorShape([fixed_batch_size.value, num_decoder_symbols]))
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, input_ in enumerate(decoder_inputs):
      # if time == 0:
      #  hidden_state = zero_state(num_decoder_symbols, batch_size)
      if time > 0: tf.get_variable_scope().reuse_variables()
      # pylint: disable=cell-var-from-loop
      # call_cell = lambda: cell(input_, state)
      generate_logit = lambda: _linear(decoder_inputs[time], num_decoder_symbols, True)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        logit = _step(
          time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit)
      else:
        logit = generate_logit
      decoder_outputs.append(logit)

  return decoder_outputs


def seq_classification_decoder_linear(num_decoder_symbols, encoder_outputs, decoder_inputs,
                                      att_initial_state, batch_size, loop_function=None,
                                      sequence_length=None, scope=None):
  """

  Simple linear decoder,


  :param num_decoder_symbols:
  :param decoder_inputs:
  :param loop_function:
  :param sequence_length:
  :param scope:
  :return:
  """
  with tf.variable_scope("seq_classification_decoder_linear", reuse=None) as scope:
    output_size = encoder_outputs[0].get_shape()[1].value
    attention = get_vinyals_attention_function(encoder_outputs, output_size, 1, scope=scope)
    attn_weights, attns = attention(att_initial_state)

    # with variable_scope.variable_scope(scope or "Linear"):
    output = _linear(attns[0], num_decoder_symbols, True)

    return [output], attn_weights
