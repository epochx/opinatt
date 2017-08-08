#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://github.com/OlavHN/bnlstm

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple, BasicLSTMCell
from tensorflow.contrib.layers import dropout


class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(1, [x, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units], dtype=tf.float32)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, 'xh', self.training)
            bn_hh = batch_norm(hh, 'hh', self.training)

            #print(bn_xh)
            #print(bn_hh)
            #print(bias)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(hidden, 4, axis=1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm(new_c, 'c', self.training)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', shape=[size],
                                initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        offset = tf.get_variable('offset', shape=[size])

        pop_mean = tf.get_variable('pop_mean', shape=[size],
                                   initializer=tf.zeros_initializer(dtype=tf.float32),
                                   trainable=False)
        pop_var = tf.get_variable('pop_var', shape=[size],
                                  initializer=tf.ones_initializer(),
                                  trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


# This is a minimal gist of what you'd have to
# add to TensorFlow code to implement zoneout.

# To see this in action, see zoneout_seq2seq.py

z_prob_cells = 0.05
z_prob_states = 0


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, zoneout_prob, is_training=True):
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if isinstance(cell, BasicLSTMCell):
      self._tuple = lambda x: LSTMStateTuple(*x)
    else:
      self._tuple = lambda x: tuple(x)
    if (isinstance(zoneout_prob, float) and
          not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
    self._cell = cell
    self._zoneout_prob = zoneout_prob
    self.is_training = is_training


  @property
  def state_size(self):
    return self._cell.state_size


  @property
  def output_size(self):
    return self._cell.output_size


  def __call__(self, inputs, state, scope=None):
    if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
      raise TypeError("Subdivided states need subdivided zoneouts.")
    if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
      raise ValueError("State and zoneout need equally many parts.")
    output, new_state = self._cell(inputs, state, scope)
    if isinstance(self.state_size, tuple):
      if self.is_training:
        new_state = self._tuple([(1 - state_part_zoneout_prob) * dropout(
          new_state_part - state_part, (1 - state_part_zoneout_prob)) + state_part
                          for new_state_part, state_part, state_part_zoneout_prob in
                          zip(new_state, state, self._zoneout_prob)])
      else:
        new_state = self._tuple([state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                          for new_state_part, state_part, state_part_zoneout_prob in
                          zip(new_state, state, self._zoneout_prob)])
    else:
      if self.is_training:
        new_state = (1 - state_part_zoneout_prob) * dropout(
          new_state_part - state_part, (1 - state_part_zoneout_prob)) + state_part
      else:
        new_state = state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
    return output, new_state

# # Wrap your cells like this
# cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=random_uniform(), state_is_tuple=True),
# zoneout_prob=(z_prob_cells, z_prob_states))