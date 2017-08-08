#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
import tensorflow as tf


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit,
                                                                   labels=target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss_by_batch(logits, targets, weights, average_across_timesteps=True,
                           softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed (averaged).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss_by_batch"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
      logits, targets, weights,
      average_across_timesteps=average_across_timesteps,
      softmax_loss_function=softmax_loss_function))
    batch_size = array_ops.shape(targets[0])[0]
    return cost / math_ops.cast(batch_size, dtypes.float32)


def get_sequence_loss(logits, targets, weights, softmax_loss_function=None, per_example_loss=False):

  if per_example_loss:
    assert len(logits) == len(targets)
    # We need to make target and int64-tensor and set its shape.
    bucket_target = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets]
    crossent = sequence_loss_by_example(logits, bucket_target, weights,
                                              softmax_loss_function=softmax_loss_function)
  else:
    assert len(logits) == len(targets)
    bucket_target = [array_ops.reshape(math_ops.to_int64(x), [-1]) for x in targets]
    crossent = sequence_loss_by_batch(logits, bucket_target, weights,
                                      softmax_loss_function=softmax_loss_function)

  return crossent


def get_classification_loss(logits, targets, softmax_loss_function=None):
  bucket_outputs = logits
  if softmax_loss_function is None:
    assert len(bucket_outputs) == len(targets) == 1
    # We need to make target an int64-tensor and set its shape.
    bucket_target = array_ops.reshape(math_ops.to_int64(targets[0]), [-1])
    crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=bucket_outputs[0],
                                                               labels=bucket_target)
  else:
    assert len(bucket_outputs) == len(targets) == 1
    crossent = softmax_loss_function(bucket_outputs[0], targets[0])

  batch_size = array_ops.shape(targets[0])[0]
  loss = tf.reduce_sum(crossent) / math_ops.cast(batch_size, dtypes.float32)

  return loss
