"""Tests for metrics.classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import classification
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class MAPETest(test.TestCase):

  def setUp(self):
    super(MAPETest, self).setUp()
    np.random.seed(1)

  def testVars(self):
    classification.mean_absolute_percentage_error(
      labels = array_ops.ones((10, 1)),
      predictions = array_ops.ones((10, 1)),
    )
    expected = {'mape/total:0', 'mape/count:0'}
    self.assertEqual(expected, {v.name for v in variables.local_variables()})
    self.assertEqual(expected, {v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)})

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mape, _ = classification.mean_absolute_percentage_error(
      labels = array_ops.ones((10, 1)),
      predictions = array_ops.ones((10, 1)),
      metrics_collections=[my_collection_name]
    )
    self.assertEqual([mape], ops.get_collection(my_collection_name))

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, mape_op = classification.mean_absolute_percentage_error(
        labels = array_ops.ones((10, 1)),
        predictions = array_ops.ones((10, 1)),
        updates_collections=[my_collection_name]
    )
    self.assertEqual([mape_op], ops.get_collection(my_collection_name))

  def testMAPEUnweighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))

    pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    mape, mape_op = classification.mean_absolute_percentage_error(labels, pred)
    with self.cached_session() as sess:
      sess.run(variables.local_variables_initializer())
      sess.run([mape_op])
      self.assertAllClose(35e7, mape.eval(), atol=1e-5)

  def testMAPEWeighted(self):
    labels = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                 (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    mape, mape_op = classification.mean_absolute_percentage_error(labels, pred, sample_weight)
    with self.cached_session() as sess:
      sess.run(variables.local_variables_initializer())
      sess.run([mape_op])
      self.assertAllClose(40e7, mape.eval(), atol=1e-5)

  def testMultiUpdates(self):
    num_samples = 6400
    batch_size = 64
    num_step = int(num_samples / batch_size)
    labels = np.random.randint(0, 1000, size=(num_samples, 1)).astype(np.float32)
    predictions = np.random.randint(0, 1000, size=(num_samples, 1)).astype(np.float32)
    expected = np.mean(
      100 * np.abs(
        (labels - predictions) \
        / np.clip(np.abs(labels), classification.EPSILON, np.inf)
      )
    )
    tf_predictions, tf_labels = dataset_ops.make_one_shot_iterator(
      dataset_ops.Dataset
      .from_tensor_slices((predictions, labels))
      .repeat()
      .batch(batch_size)).get_next()
    mape, mape_op = classification.mean_absolute_percentage_error(tf_labels, tf_predictions)
    with self.cached_session() as sess:
      sess.run(variables.local_variables_initializer())
      for _ in range(num_step):
        sess.run([mape_op])
      self.assertAllClose(expected, mape.eval(), atol=1e-5)

if __name__ == '__main__':
    test.main()
