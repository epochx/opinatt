#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from .models import dep_attention_RNN_linear
from .utils import _extract_argmax_and_one_hot, PAD_ID, contextwin
from .losses import get_classification_loss, get_sequence_loss
from .cells import ZoneoutWrapper, BNLSTMCell


class MultiTaskModel(object):

  def __init__(self, source_vocab_size, tag_vocab_size, label_vocab_size, buckets,
               word_embedding_size, size, num_layers, max_gradient_norm, batch_size,
               dropout_keep_prob=1.0, rnn="lstm", bidirectional_rnn=True,
               use_attention=False, task=None, forward_only=False, context_win_size=1,
               word_embedding_matrix=None, train_embeddings=True, use_previous_output=True,
               learning_rate=0.1, decay_factor=1.0, use_binary_features=False,
               binary_feat_dim=14, optimizer="sgd", zoneout_keep_prob=1.0):

    if word_embedding_size == 0 and word_embedding_matrix is not None:
      assert word_embedding_matrix.shape[0] == source_vocab_size
      word_embedding_size = word_embedding_matrix.shape[1]

    self.source_vocab_size = source_vocab_size
    self.tag_vocab_size = tag_vocab_size
    self.label_vocab_size = label_vocab_size
    self.context_win_size = context_win_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.use_binary_features = use_binary_features
    self.binary_feat_dim = binary_feat_dim
    self.global_step = tf.Variable(0, trainable=False)

    if decay_factor < 1:
      self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                      100000, decay_factor, staircase=True)
    else:
      self.learning_rate = learning_rate

    # Create the internal multi-layer cell for our RNN.
    if rnn == "lstm":
      single_cell = tf.contrib.rnn.BasicLSTMCell(size)
    elif rnn == "gru":
      single_cell = tf.contrib.rnn.GRUCell(size)
    elif rnn == "bnlstm":
      single_cell = BNLSTMCell(size, tf.cast(not forward_only, tf.bool))
    cell = single_cell
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

    if not forward_only and dropout_keep_prob < 1.0:
      cell = tf.contrib.rnn.DropoutWrapper(cell,
                                           input_keep_prob=dropout_keep_prob,
                                           output_keep_prob=dropout_keep_prob)

    if zoneout_keep_prob < 1.0:
      cell = ZoneoutWrapper(cell, (zoneout_keep_prob, zoneout_keep_prob),
                            is_training=not forward_only)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.tags = []
    self.tag_weights = []
    self.labels = []
    self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
    self.encoder_extra_inputs = []

    for i in xrange(buckets[-1][0]):
      if context_win_size > 1:
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=None, name="encoder{0}".format(i)))
      else:
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

      self.encoder_extra_inputs.append(tf.placeholder(tf.float32, name="encoder_extra{0}".format(i)))

    for i in xrange(buckets[-1][1]):
      self.tags.append(tf.placeholder(tf.int32, shape=[None], name="tag{0}".format(i)))
      self.tag_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    self.labels.append(tf.placeholder(tf.float32, shape=[None], name="label"))

    if use_previous_output:
      loop_function = _extract_argmax_and_one_hot(self.tag_vocab_size)
    else:
      loop_function = None

    output = dep_attention_RNN_linear(self.encoder_inputs,
                                      self.encoder_extra_inputs,
                                      self.tags,
                                      cell,
                                      self.source_vocab_size,
                                      self.tag_vocab_size,
                                      self.label_vocab_size,
                                      word_embedding_size,
                                      self.batch_size,
                                      task,
                                      binary_feat_dim=0,
                                      dtype=tf.float32,
                                      loop_function=loop_function,
                                      sequence_length=self.sequence_length,
                                      bidirectional_rnn=bidirectional_rnn,
                                      context_win_size=context_win_size,
                                      train_embeddings=train_embeddings)

    seq_output, seq_attention, class_output, class_attention = output

    if task['tagging'] == 1:
      self.tagging_output = seq_output
      self.attention_output = seq_attention
      self.tagging_loss = get_sequence_loss(seq_output, self.tags, self.tag_weights,
                                            per_example_loss=forward_only)

    if task['intent'] == 1:
      self.classification_output = class_output
      self.classification_loss = get_classification_loss(class_output, self.labels)
      self.classification_attention = class_attention

    if task['tagging'] == 1:
      self.loss = self.tagging_loss
    elif task['intent'] == 1:
      self.loss = self.classification_loss

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      if optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif optimizer == "adam":
        opt = tf.train.AdamOptimizer()

      if task['joint'] == 1:
        # backpropagate the intent and tagging loss, one may further adjust
        # the weights for the two costs.
        gradients = tf.gradients([self.tagging_loss, self.classification_loss], params)
      elif task['tagging'] == 1:
        gradients = tf.gradients(self.tagging_loss, params)
      elif task['intent'] == 1:
        gradients = tf.gradients(self.classification_loss, params)

      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norm = norm
      self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                        global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())

  def joint_step(self, session, encoder_inputs, encoder_extra_inputs, tags, tag_weights,
                 labels, batch_sequence_length, bucket_id, forward_only):
    """Run a step of the joint model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      encoder_extra_inputs: losf of binary features to feed as encoder extra features
      tags: list of numpy int vectors to feed as decoder inputs.
      tag_weights: list of numpy float vectors to feed as tag weights.
      labels: list of numpy int vectors to feed as sequence class labels.
      bucket_id: which bucket of the model to use.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, output tags, and output class label.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))
    if len(labels) != 1:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(labels), 1))

    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]
      if self.use_binary_features:
        input_feed[self.encoder_extra_inputs[l].name] = encoder_extra_inputs[l]
    input_feed[self.labels[0].name] = labels[0]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss]  # Loss for this batch.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      for i in range(tag_size):
        output_feed.append(self.attention_output[i])

      output_feed.append(self.classification_output[0])
      output_feed.append(self.classification_attention[0])
    else:
      output_feed = [self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
      for i in range(tag_size):
        output_feed.append(self.attention_output[i])

      output_feed.append(self.classification_output[0])
      output_feed.append(self.classification_attention[0])

    outputs = session.run(output_feed, input_feed)
    # if not forward_only:
    #  return outputs[1], outputs[2], outputs[3:3+tag_size], outputs[-1]
    # else:
    #  return None, outputs[0], outputs[1:1+tag_size], outputs[-1]
    if not forward_only:
      return (outputs[1], outputs[2], outputs[3:3 + tag_size],
              outputs[3 + tag_size:3 + 2 * tag_size],
              outputs[-2], outputs[-1])
    else:
      return (None, outputs[0], outputs[1:1 + tag_size],
              outputs[1 + tag_size:1 + 2 * tag_size],
              outputs[-2], outputs[-1])

  def tagging_step(self, session, encoder_inputs, encoder_extra_inputs, tags,
                   tag_weights, batch_sequence_length, bucket_id, forward_only):
    """Run a step of the tagging model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      encoder_extra_inputs: losf of binary features to feed as encoder extra features
      tags: list of numpy int vectors to feed as decoder inputs.
      tag_weights: list of numpy float vectors to feed as target weights.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the output tags.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]
      if self.use_binary_features:
        input_feed[self.encoder_extra_inputs[l].name] = encoder_extra_inputs[l]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss]  # Loss for this batch.
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])

      for i in range(tag_size):
        output_feed.append(self.attention_output[i])

    else:
      output_feed = [self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])

      for i in range(tag_size):
        output_feed.append(self.attention_output[i])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3 + tag_size], outputs[3 + tag_size:3 + 2 * tag_size]
    else:
      return None, outputs[0], outputs[1:1 + tag_size], outputs[1 + tag_size:1 + 2 * tag_size]

  def classification_step(self, session, encoder_inputs, encoder_extra_inputs,
                          labels, batch_sequence_length, bucket_id, forward_only):
    """Run a step of the intent classification model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      encoder_extra_inputs: losf of binary features to feed as encoder extra features
      labels: list of numpy int vectors to feed as sequence class labels.
      batch_sequence_length: batch_sequence_length
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the output class label.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, target_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      if self.use_binary_features:
        input_feed[self.encoder_extra_inputs[l].name] = encoder_extra_inputs[l]
    input_feed[self.labels[0].name] = labels[0]

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,  # Update Op that does SGD.
                     self.gradient_norm,  # Gradient norm.
                     self.loss,  # Loss for this batch.
                     self.classification_output[0]]
    else:
      output_feed = [self.loss,
                     self.classification_output[0], ]

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, outputs.
    else:
      return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.


  def get_batch(self, data, bucket_id, sample_start_id=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, encoder_extra_inputs, decoder_inputs, labels = [], [], [], []
    context_encoder_inputs = []
    counter = None
    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed and add GO to decoder.
    batch_sequence_length_list = list()
    if sample_start_id is not None:
      counter = sample_start_id

    for _ in xrange(self.batch_size):
      if counter is not None:
        encoder_input, encoder_extra_input, decoder_input, label = data[bucket_id][counter]
        counter += 1
      else:
        encoder_input, encoder_extra_input, decoder_input, label = random.choice(data[bucket_id])

      batch_sequence_length_list.append(len(encoder_input))

      # padding encoder inputs
      pad_size = encoder_size - len(encoder_input)
      encoder_pad = [PAD_ID] * pad_size
      encoder_input = encoder_input + encoder_pad
      encoder_inputs.append(list(encoder_input))

      # Adding context windows
      if self.context_win_size > 1:
        context_encoder_input = contextwin(encoder_input, self.context_win_size)
        context_encoder_inputs.append(list(context_encoder_input))

      if self.use_binary_features:
        zero_binary_feats = [0] * self.binary_feat_dim
        padded_encoder_extra_input = encoder_extra_input + [zero_binary_feats] * pad_size
      else:
        padded_encoder_extra_input = []

      encoder_extra_inputs.append(list(padded_encoder_extra_input))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append(decoder_input + [PAD_ID] * decoder_pad_size)
      labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels = [], [], [], []
    batch_encoder_extra_inputs = []

    if self.context_win_size > 1:
      batch_context_encoder_inputs = []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):

      batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                            for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      if self.context_win_size > 1:
        batch_context_encoder_input = [context_encoder_inputs[batch_idx][length_idx]
                                       for batch_idx in xrange(self.batch_size)]
        batch_context_encoder_inputs.append(np.asarray(batch_context_encoder_input, dtype=np.int32))

      if self.use_binary_features:
        batch_encoder_extra_input = [encoder_extra_inputs[batch_idx][length_idx]
                                     for batch_idx in xrange(self.batch_size)]

        batch_encoder_extra_inputs.append(np.asarray(batch_encoder_extra_input, dtype=np.float32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
        np.array([decoder_inputs[batch_idx][length_idx]
                  for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        #        if length_idx < decoder_size - 1:
        #          target = decoder_inputs[batch_idx][length_idx + 1]
        #        print (length_idx)
        if decoder_inputs[batch_idx][length_idx] == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    batch_labels.append(
      np.array([labels[batch_idx]
                for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)

    if self.context_win_size > 1:
      return batch_context_encoder_inputs, batch_encoder_inputs, batch_encoder_extra_inputs, \
             batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels
    else:
      return batch_encoder_inputs, batch_encoder_inputs, batch_encoder_extra_inputs, \
             batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels

  def get_one(self, data, bucket_id, sample_id):
    """Get a single sample data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, encoder_extra_inputs, decoder_inputs, labels = [], [], [], []
    context_encoder_inputs = []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_sequence_length_list = list()
    # for _ in xrange(self.batch_size):
    encoder_input, encoder_extra_input, decoder_input, label = data[bucket_id][sample_id]
    batch_sequence_length_list.append(len(encoder_input))

    # Encoder inputs are padded
    pad_size = encoder_size - len(encoder_input)
    encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
    encoder_input = encoder_input + encoder_pad
    encoder_inputs.append(list(encoder_input))

    if self.context_win_size > 1:
      context_encoder_input = contextwin(encoder_input, self.context_win_size)
      context_encoder_inputs.append(list(context_encoder_input))

    if self.use_binary_features:
      zero_binary_feats = [0] * self.binary_feat_dim
      padded_encoder_extra_input = encoder_extra_input + [zero_binary_feats] * pad_size

    else:
      padded_encoder_extra_input = []

    encoder_extra_inputs.append(list(padded_encoder_extra_input))

    # Decoder inputs get an extra "GO" symbol, and are padded then.
    decoder_pad_size = decoder_size - len(decoder_input)
    decoder_inputs.append(decoder_input + [PAD_ID] * decoder_pad_size)
    labels.append(label)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels = [], [], [], []
    batch_encoder_extra_inputs = []

    if self.context_win_size > 1:
      batch_context_encoder_inputs = []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                            for batch_idx in xrange(1)], dtype=np.int32))
      if self.context_win_size > 1:
        batch_context_encoder_input = [context_encoder_inputs[batch_idx][length_idx]
                                       for batch_idx in xrange(1)]
        batch_context_encoder_inputs.append(np.asarray(batch_context_encoder_input, dtype=np.int32))

      if self.use_binary_features:
        batch_encoder_extra_input = [encoder_extra_inputs[batch_idx][length_idx]
                                     for batch_idx in xrange(1)]

        batch_encoder_extra_inputs.append(np.asarray(batch_encoder_extra_input, dtype=np.float32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
        np.array([decoder_inputs[batch_idx][length_idx]
                  for batch_idx in xrange(1)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(1, dtype=np.float32)
      for batch_idx in xrange(1):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        #        if length_idx < decoder_size - 1:
        #          target = decoder_inputs[batch_idx][length_idx + 1]
        #        print (length_idx)
        if decoder_inputs[batch_idx][length_idx] == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    batch_labels.append(
      np.array([labels[batch_idx][0]
                for batch_idx in xrange(1)], dtype=np.int32))

    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)

    if self.context_win_size > 1:
      return batch_context_encoder_inputs, batch_encoder_inputs, batch_encoder_extra_inputs, \
             batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels
    else:
      return batch_encoder_inputs, batch_encoder_inputs, batch_encoder_extra_inputs, \
             batch_decoder_inputs, batch_weights, batch_sequence_length, batch_labels