"""Classification metrics library"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl

EPSILON = 1e-7

def mean_absolute_percentage_error(labels,
                        predictions,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """Computes the mean absolute percentage error between the labels and predictions.

  The `mean_absolute_percentage_error` function creates two local variables,
  `total` and `count` that are used to compute the mean absolute percentage error.
  This average is weighted by `weights`, and it is ultimately returned as
  `mean_absolute_percentage_error`: an idempotent operation that simply divides `total`
  by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_absolute_percentage_error`. Internally, an `absolute_percentage_errors` operation
  computes the absolute value of the percentage differences between `predictions` and `labels`.
  Then `update_op` increments `total` with the reduced sum of the product of
  `weights` and `absolute_percentage_errors`, and it increments `count` with the reduced
  sum of `weights`

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` of arbitrary shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_absolute_percentage_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_absolute_percentage_error: A `Tensor` representing the current mean, the value
    of `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_absolute_percentage_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_absolute_percentage_error is not supported '
                       'when eager execution is enabled.')

  if predictions.dtype in (dtypes.float16, dtypes.float32, dtypes.float64) \
      and labels.dtype != predictions.dtype:
    labels = math_ops.cast(labels, predictions.dtype)
  elif labels.dtype in (dtypes.float16, dtypes.float32, dtypes.float64) \
      and labels.dtype != predictions.dtype:
    predictions = math_ops.cast(predictions, labels.dtype)
  else:
    labels = math_ops.cast(labels, dtypes.float32)
    predictions = math_ops.cast(predictions, dtypes.float32)

  predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  min_value = constant_op.constant(EPSILON, dtype=dtypes.float32)
  max_value = constant_op.constant(float('Inf'), dtype=dtypes.float32)
  percentage_absolute_errors = 100 * math_ops.abs(
      (predictions - labels) / math_ops.abs(clip_ops.clip_by_value(math_ops.abs(labels), min_value, max_value)))
  return metrics_impl.mean(percentage_absolute_errors, weights, metrics_collections,
              updates_collections, name or 'mape')
