import tensorflow as tf
import glob
from model import SequenceAutoencoder
from model import LanguageModel

from dataset import DatasetBuilder
from commons import tokenization
from commons import layers
import os

def main():
  method = 'sa'
  batch_size = 64
  shuffle = True
  max_length = 500 if method == 'sa' else None
  num_parallel_calls = 8
  num_buckets = 10
  bucket_width = 100
  random_seed = 42
  vocab_path = 'vocab'
  hidden_size = 512
  dropout_rate = 0.2
  clip_norm = 5.0


  builder = DatasetBuilder(batch_size=batch_size,
                           shuffle=shuffle,
                           max_length=max_length,
                           num_parallel_calls=num_parallel_calls,
                           num_buckets=num_buckets,
                           bucket_width=bucket_width,
                           random_seed=random_seed)

  filenames = sorted(glob.glob('unlabeled/*.tfrecord'))
  dataset = builder.build_pretrain_dataset(filenames)

  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size

  if method == 'sa':
    model = SequenceAutoencoder(
        vocab_size, hidden_size, dropout_rate=dropout_rate, dropout_embedding=True)
  else:
    model = LanguageModel(
        vocab_size, hidden_size, dropout_rate=dropout_rate, dropout_embedding=True)


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


  ckpt_path = method
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  latest_ckpt = tf.train.latest_checkpoint(ckpt_path)

  if latest_ckpt:
    print('Restoring from checkpoint: %s ...' % latest_ckpt)
    ckpt.restore(latest_ckpt)
  else:
    print('Training from scratch...')


  def split(token_ids, s=100):
    l = tf.shape(token_ids)[1]

    boundaries = tf.concat([tf.range(l, delta=s), [l]], axis=0)
    splits = boundaries[1:] - boundaries[:-1]
    return tf.split(token_ids, splits, axis=1)


  if method == 'sa':
    for token_ids in dataset:
      loss, step, lr = train_step(token_ids)
      if step.numpy() % 100 == 0:
        print(step.numpy(), loss.numpy(), lr.numpy())

      if step.numpy() % 10000 == 0:
        ckpt.save(os.path.join(ckpt_path, 'model'))

      if step.numpy() == 200000:
        break
  else:
    for i, token_ids in enumerate(dataset):
      inputs = token_ids[:, :-1]
      labels = token_ids[:, 1:] 

      inputs_list = split(inputs)
      labels_list = split(labels)

      states = tf.zeros((batch_size, hidden_size)), tf.zeros((batch_size, hidden_size))


      for x, y in zip(inputs_list, labels_list):
        loss, states, step, lr = train_step(x, y, states)
        
      if i % 100 == 0:
        print(i, loss.numpy(), lr.numpy())
        
      if i % 10000 == 0:
        ckpt.save(os.path.join(ckpt_path, 'model'))

      if i >= 120000:
        break
main()
