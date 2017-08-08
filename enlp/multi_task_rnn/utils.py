#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
#from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.util import nest
import tensorflow as tf
from enlp.settings import PAD_ID


def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.
  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.
  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.
  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term







def contextwin(l, win, pad_id=PAD_ID):
  '''
  win :: int corresponding to the size of the window
  given a list of indexes composing a sentence
  it will return a list of list of indexes corresponding
  to context windows surrounding each word in the sentence
  '''
  assert (win % 2) == 1
  assert win >= 1
  l = list(l)

  lpadded = win // 2 * [pad_id] + l + win // 2 * [pad_id]
  out = [lpadded[i:i + win] for i in range(len(l))]

  assert len(out) == len(l)

  return out


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def _extract_argmax_and_one_hot(one_hot_size,
                                output_projection=None):
  """Get a loop_function that extracts the previous symbol and build a one-hot vector for it.

  Args:
    one_hot_size: total size of one-hot vector.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.one_hot(prev_symbol, one_hot_size)
    return emb_prev

  return loop_function


def _step(time, sequence_length, min_sequence_length, max_sequence_length, zero_logit, generate_logit):
  # Step 1: determine whether we need to call_cell or not
  empty_update = lambda: zero_logit
  logit = control_flow_ops.cond(
    time < max_sequence_length, generate_logit, empty_update)

  # Step 2: determine whether we need to copy through state and/or outputs
  existing_logit = lambda: logit

  def copy_through():
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    copy_cond = (time >= sequence_length)
    return math_ops.select(copy_cond, zero_logit, logit)

  logit = control_flow_ops.cond(
    time < min_sequence_length, existing_logit, copy_through)
  logit.set_shape(logit.get_shape())
  return logit


def zero_state(state_size, batch_size, dtype):
  """Return zero-filled state tensor(s).
  Args:
    batch_size: int, float, or unit Tensor representing the batch size.
    dtype: the data type to use for the state.
  Returns:
    If `state_size` is an int or TensorShape, then the return value is a
    `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
    If `state_size` is a nested list or tuple, then the return value is
    a nested list or tuple (of the same structure) of `2-D` tensors with
  the shapes `[batch_size x s]` for each s in `state_size`.
  """
  zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
  zeros = tf.zeros(tf.stack(zeros_size), dtype=dtype)
  zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))
  return zeros



