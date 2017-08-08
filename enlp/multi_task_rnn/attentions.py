#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.python.ops import nn_ops
from .utils import _linear
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def get_vinyals_attention_function(attention_inputs, output_size, num_heads, scope=None):
  """
  This is the attention approach as defined by Vinyals and Kaiser on
  the paer 'Grammar as a Foreign Language'

  :param attention_inputs:
  :param output_size:
  :param num_heads:
  :return:
  """
  with tf.variable_scope(scope or "vinyals_attention", reuse=None) as attention_scope:

    top_states = [array_ops.reshape(e, [-1, 1, output_size])
                  for e in attention_inputs]
    attention_states = array_ops.concat(top_states, 1)
    if not attention_states.get_shape()[1:2].is_fully_defined():
      raise ValueError("Shape[1] and [2] of attention_states must be known: %s" % attention_states.get_shape())

    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      attn_weights = []
      ds = []  # Results of attention reads will be stored here.
      for i in xrange(num_heads):
        with tf.variable_scope("Attention_%d" % i):
          y = _linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[i] * math_ops.tanh(hidden_features[i] + y), [2, 3])
          a = nn_ops.softmax(s)
          attn_weights.append(a)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return attn_weights, ds

    return attention

