"""Pipeline for pretraining a Sequence Autoencoder or Language Model for 
LSTM-based model that classifies IMDB movie reviews.
"""
import glob
import os

import tensorflow as tf
from absl import flags
from absl import app

from commons import layers
from commons import tokenization
from commons.dataset import SequenceClassifierDatasetBuilder 
from model import LanguageModel
from model import SequenceAutoencoder
import utils


flags.DEFINE_enum(
    'method', 'sa', ['sa', 'lm'], 'Method for pretraining: Sequence Autoencoder'
        ' or Language Model.')
flags.DEFINE_string(
    'data_dir', None, 'Path to the directory holding *.tfrecord files') 
flags.DEFINE_integer(
    'batch_size', 64, 'Static batch size')
flags.DEFINE_bool(
    'shuffle', True, 'Whether to shuffle training data.')
flags.DEFINE_integer(
    'max_length', 500, 'Source or target seqs longer than this will be filtered'
        ' out (Only applicable when "method" is "sa").')
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
flags.DEFINE_float(
    'clip_norm', 5.0, 'The value that the norm of gradient will be clipped to.')


FLAGS = flags.FLAGS

def main(_):
  method = FLAGS.method
  data_dir = FLAGS.data_dir
  batch_size = FLAGS.batch_size
  shuffle = FLAGS.shuffle
  max_length = FLAGS.max_length
  num_parallel_calls = FLAGS.num_parallel_calls
  num_buckets = FLAGS.num_buckets
  bucket_width = FLAGS.bucket_width
  vocab_path = FLAGS.vocab_path 
  hidden_size = FLAGS.hidden_size
  dropout_rate = FLAGS.dropout_rate
  clip_norm = FLAGS.clip_norm
 
  builder = SequenceClassifierDatasetBuilder(
      batch_size=batch_size,
      shuffle=shuffle,
      max_length=max_length,
      num_parallel_calls=num_parallel_calls,
      num_buckets=num_buckets,
      bucket_width=bucket_width,
      random_seed=42)

  filenames = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord')))
  dataset = builder.build_pretrain_dataset(filenames)

  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size

  if method == 'sa':
    model = SequenceAutoencoder(vocab_size, 
                                hidden_size, 
                                dropout_rate=dropout_rate, 
                                dropout_embedding=True)
  else:
    model = LanguageModel(vocab_size, 
                          hidden_size, 
                          dropout_rate=dropout_rate, 
                          dropout_embedding=True)

  optimizer = tf.keras.optimizers.Adam(
      tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries=[100000],
          values=[0.001, 0.0001]))

  if method == 'sa':
    train_step_signature = [
        tf.TensorSpec(shape=(batch_size, None), dtype='int64')]

    @tf.function(input_signature=train_step_signature)
    def train_step(token_ids):
      with tf.GradientTape() as tape:
        logits = model(token_ids)
        loss = layers.compute_loss(token_ids, logits, 0.1, vocab_size)

      gradients = tape.gradient(loss, model.trainable_variables)
      if clip_norm is not None:
        grdients, norm = tf.clip_by_global_norm(gradients, clip_norm)

      optimizer.apply_gradients(
          zip(gradients, model.trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step) 
      return loss, step - 1, lr
  else:
    train_step_signature = [
      tf.TensorSpec(shape=(batch_size, None), dtype='int64'),
      tf.TensorSpec(shape=(batch_size, None), dtype='int64'),
          (tf.TensorSpec(shape=(batch_size, hidden_size), dtype='float32'),
           tf.TensorSpec(shape=(batch_size, hidden_size), dtype='float32'))]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, labels, states):
      with tf.GradientTape() as tape:
        logits, new_states = model(inputs, states)
        loss = layers.compute_loss(labels, logits, 0.1, vocab_size)

      gradients = tape.gradient(loss, model.trainable_variables)
      if clip_norm is not None:
        grdients, norm = tf.clip_by_global_norm(gradients, clip_norm)

      optimizer.apply_gradients(
          zip(gradients, model.trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return loss, new_states, step - 1, lr

  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  print('Training from scratch...')

  if method == 'sa':
    for i, token_ids in enumerate(dataset):
      loss, step, lr = train_step(token_ids)

      if i % 100 == 0:
        print(i, loss.numpy(), lr.numpy())

      if i % 10000 == 0:
        ckpt.save(os.path.join(method, 'model'))

      if i == 120000:
        break
  else:
    for i, token_ids in enumerate(dataset):
      inputs = token_ids[:, :-1]
      labels = token_ids[:, 1:] 

      inputs_list = utils.split(inputs)
      labels_list = utils.split(labels)

      states = (tf.zeros((batch_size, hidden_size)), 
                tf.zeros((batch_size, hidden_size)))

      for x, y in zip(inputs_list, labels_list):
        loss, states, step, lr = train_step(x, y, states)
        
      if i % 100 == 0:
        print(i, loss.numpy(), lr.numpy())
        
      if i % 10000 == 0:
        ckpt.save(os.path.join(method, 'model'))

      if i >= 120000:
        break

if __name__ == '__main__':
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('vocab_path')

  app.run(main)
