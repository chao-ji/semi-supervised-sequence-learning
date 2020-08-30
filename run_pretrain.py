import tensorflow as tf
import glob
from model import SequenceAutoencoder
from dataset import DatasetBuilder
import tokenization
from commons import layers
import os

def main():
  batch_size = 64
  shuffle = True
  max_length = 500
  num_parallel_calls = 8
  num_buckets = 10
  bucket_width = 100
  random_seed = 42
  vocab_path = 'vocab'
  hidden_size = 512

  builder = DatasetBuilder(batch_size=batch_size,
                           shuffle=shuffle,
                           max_length=max_length,
                           num_parallel_calls=num_parallel_calls,
                           num_buckets=num_buckets,
                           bucket_width=bucket_width,
                           random_seed=random_seed)

  filenames = sorted(glob.glob('unlabeled/*.tfrecord'))
  dataset = builder.build_dataset(filenames)



  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size
  model = SequenceAutoencoder(vocab_size, hidden_size)



  clip_norm = 5.0 #None


  optimizer = tf.keras.optimizers.Adam()


  train_step_signature = [
      tf.TensorSpec(shape=(batch_size, None), dtype='int64')]

  @tf.function(input_signature=train_step_signature)
  def train_step(token_ids):
    with tf.GradientTape() as tape:
      logits = model(token_ids)
      loss = layers.compute_loss(token_ids[:, 1:], logits, 0.1, vocab_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    if clip_norm is not None:
      grdients, norm = tf.clip_by_global_norm(gradients, clip_norm)

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

    step = optimizer.iterations

    return loss, step - 1

  ckpt_path = 'sa'
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  latest_ckpt = tf.train.latest_checkpoint(ckpt_path)

  if latest_ckpt:
    print('Restoring from checkpoint: %s ...' % latest_ckpt)
    ckpt.restore(latest_ckpt)
  else:
    print('Training from scratch...')


  for token_ids in dataset:
    loss, step = train_step(token_ids)
    if step.numpy() % 100 == 0:
      print(step.numpy(), loss.numpy())

    if step.numpy() % 10000 == 0:
      ckpt.save(os.path.join(ckpt_path, 'model'))

    if step.numpy() == 200000:
      break



main()
