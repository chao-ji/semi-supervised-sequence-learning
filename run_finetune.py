"""Pipeline for fine-tuning sequence classifier on labeled IMDB movie review 
dataset.
"""
import glob
import itertools
import os

import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from commons import tokenization
from commons.dataset import SequenceClassifierDatasetBuilder 
from model import SequenceClassifier


flags.DEFINE_enum(
    'method', 'sa', ['sa', 'lm'], 'Method for pretraining: Sequence Autoencoder'
        ' or Language Model.')
flags.DEFINE_string(
    'data_dir', None, 'Path to the directory holding *.tfrecord files')
flags.DEFINE_string(
    'test_pos_filename', None, 'Path to the name of the file storing positive '
        'examples in the test set.')
flags.DEFINE_string(
    'test_neg_filename', None, 'Path to the name of the file storing negative '
        'examples in the test set.')
flags.DEFINE_bool(
    'no_finetune', False, 'Finetune on pretrained weights (False) or train '
        'from scratch (True).')
flags.DEFINE_integer( 
    'batch_size', 64, 'Static batch size')
flags.DEFINE_bool(
    'shuffle', True, 'Whether to shuffle training data.')
flags.DEFINE_integer(
    'num_parallel_calls', 8, 'Num of TFRecord files to be processed '
        'concurrently.')
flags.DEFINE_integer(
    'num_buckets', 10, 'Number of sequence length buckets.')
flags.DEFINE_integer(
    'bucket_width', 100, 'Size of each sequence length bucket.')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_float( 
    'dropout_rate', 0.2, 'Dropout rate for the Dropout layers.')


FLAGS = flags.FLAGS

def main(_):
  method = FLAGS.method
  data_dir = FLAGS.data_dir
  test_pos_filename = FLAGS.test_pos_filename
  test_neg_filename = FLAGS.test_neg_filename
  no_finetune = FLAGS.no_finetune
  batch_size = FLAGS.batch_size
  shuffle = FLAGS.shuffle
  num_parallel_calls = FLAGS.num_parallel_calls
  num_buckets = FLAGS.num_buckets
  bucket_width = FLAGS.bucket_width
  vocab_path = FLAGS.vocab_path
  hidden_size = FLAGS.hidden_size
  dropout_rate = FLAGS.dropout_rate

  builder = SequenceClassifierDatasetBuilder(
      batch_size=batch_size,
      shuffle=shuffle,
      max_length=None,
      num_parallel_calls=num_parallel_calls,
      num_buckets=num_buckets,
      bucket_width=bucket_width,
      random_seed=42)

  filenames = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord')))
  dataset = builder.build_finetune_dataset(filenames)

  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size

  model = SequenceClassifier(
      vocab_size, hidden_size, dropout=dropout_rate, dropout_embedding=True) 

  if not no_finetune:
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(method)
    if latest_ckpt is not None:
      print('Loaded latest checkpoint ', latest_ckpt)
    else:
      raise ValueError('No checkpoint found!')
    ckpt.restore(latest_ckpt).expect_partial()

  optimizer = tf.keras.optimizers.Adam()

  train_step_signature = [
      tf.TensorSpec(shape=(batch_size, None), dtype=tf.int64),
      tf.TensorSpec(shape=(batch_size, 1), dtype=tf.int64)]

  @tf.function(input_signature=train_step_signature)
  def train_step(token_ids, labels):
    with tf.GradientTape() as tape:
      logits = model.call(token_ids, training=True)

      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(labels, 'float32'), logits=logits)
      loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))
    return loss 


  for i, (token_ids, labels) in enumerate(dataset):
    loss = train_step(token_ids, labels)
    if i % 100 == 0:
      print(i, loss)

    if i == 1200:
      break 

  data = [tf.io.gfile.GFile(fn) 
      for fn in [test_pos_filename] + [test_neg_filename]]
  data = itertools.chain(*data)

  logits = []
  for l in data:
    token_ids = np.reshape(subtokenizer.encode(l, add_eos=True), [1, -1])
    logits.append(model.call(token_ids, training=False).numpy()[0, 0])

  probs = np.array(logits) >= 0 
  groundtruths = np.arange(25000) < 12500
  test_accuracy = (probs == groundtruths).sum() / 25000
  print('test_accuracy', test_accuracy)

if __name__ == '__main__':
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('vocab_path')
  flags.mark_flag_as_required('test_pos_filename') 
  flags.mark_flag_as_required('test_neg_filename') 

  app.run(main)
