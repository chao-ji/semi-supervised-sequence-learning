""""""
import tensorflow as tf 
import glob


_READ_RECORD_BUFFER = 8 * 1000 * 1000


def _parse_example(serialized_example):
  parse_dict = {'token_ids': tf.io.VarLenFeature(tf.int64)}
  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  token_ids = tf.sparse.to_dense(parsed['token_ids'])
  return token_ids


class DatasetBuilder(object):


  def __init__(self,  
               batch_size,
               shuffle,
               max_length,
               num_parallel_calls,
               num_buckets=8,
               bucket_width=10,
               random_seed=None):
    """"""
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_length = max_length
    self._num_parallel_calls = num_parallel_calls
    self._num_buckets = num_buckets
    self._bucket_width = bucket_width
    self._random_seed = random_seed

  def build_dataset(self, filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
        len(filenames), seed=self._random_seed)
    options = tf.data.Options()
    options.experimental_deterministic = False if self._shuffle else True
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename, buffer_size=_READ_RECORD_BUFFER).shuffle(750),
        cycle_length=self._num_parallel_calls,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)
    dataset = dataset.map(
        _parse_example, num_parallel_calls=self._num_parallel_calls)

    if self._max_length is not None:
      dataset = dataset.filter(lambda x: tf.size(x) <= self._max_length)

    def example_to_bucket_id(token_ids):
      seq_len = tf.size(token_ids)
      bucket_id = seq_len // self._bucket_width
      return tf.cast(tf.minimum(self._num_buckets - 1, bucket_id), 'int64')

    def batching_fn(bucket_id, grouped_dataset):
      return grouped_dataset.padded_batch(
          self._batch_size, (None,), drop_remainder=True)

    dataset = dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=self._batch_size))

    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
  batch_size = 64
  shuffle = True
  max_length = 500
  num_parallel_calls = 8
  num_buckets = 10
  bucket_width = 100
  random_seed = 42

  builder = DatasetBuilder(batch_size=batch_size,
                           shuffle=shuffle,
                           max_length=max_length,
                           num_parallel_calls=num_parallel_calls,
                           num_buckets=num_buckets,
                           bucket_width=bucket_width,
                           random_seed=random_seed)

  filenames = sorted(glob.glob('unlabeled/*.tfrecord'))
  dataset = builder.build_dataset(filenames)


  it = iter(dataset)
  a = next(it)

