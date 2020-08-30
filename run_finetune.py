import tensorflow as tf
import glob
import numpy as np
from model import SequenceClassifier
import itertools
import tokenization


def _parse_example(serialized_example):

  parse_dict = {'token_ids': tf.io.VarLenFeature(tf.int64),
                'label': tf.io.VarLenFeature(tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  token_ids = tf.sparse.to_dense(parsed['token_ids'])
  label = tf.sparse.to_dense(parsed['label'])#[0]

  return token_ids, label

_READ_RECORD_BUFFER = 8 * 1000 * 1000
shuffle = True
num_parallel_calls = 8
random_seed = 0
filenames = sorted(glob.glob('labeled/*.tfrecord'))

dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
    len(filenames), seed=random_seed)


options = tf.data.Options()
options.experimental_deterministic = False if shuffle else True
dataset = dataset.interleave(
    lambda filename: tf.data.TFRecordDataset(
        filename, buffer_size=_READ_RECORD_BUFFER).shuffle(250),
    cycle_length=num_parallel_calls,
    num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

dataset = dataset.map(
        _parse_example, num_parallel_calls=num_parallel_calls)


bucket_width = 100
num_buckets = 10
batch_size = 64 

def example_to_bucket_id(token_ids, _):
  seq_len = tf.size(token_ids)
  bucket_id = seq_len // bucket_width
  return tf.cast(tf.minimum(num_buckets - 1, bucket_id), 'int64')
  

def batching_fn(bucket_id, grouped_dataset):
  return grouped_dataset.padded_batch(
      batch_size, ([None], [1]), drop_remainder=True)

dataset = dataset.apply(tf.data.experimental.group_by_window(
    key_func=example_to_bucket_id,
    reduce_func=batching_fn,
    window_size=batch_size))
dataset = dataset.repeat(-1)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

vocab_path = 'vocab'
subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)

vocab_size = subtokenizer.vocab_size

hidden_size = 512 

model = SequenceClassifier(vocab_size, hidden_size) 

ckpt_path = 'sa' #'sa'
ckpt = tf.train.Checkpoint(model=model)
latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
print('Loaded latest checkpoint ', latest_ckpt)
ckpt.restore(latest_ckpt).expect_partial()

#dummy_inputs = tf.constant(np.random.randint(0, vocab_size, (batch_size, 18)))
#dummy_padding_mask = dummy_inputs == 0
#weights = np.load('lm_weights2.npy', allow_pickle=True)
#model(dummy_inputs, dummy_padding_mask)
#model._embedding_logits_layer.set_weights([weights[0]])
#model._recurrent_layer.set_weights(weights[1:4]) 


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
  myloss = train_step(token_ids, labels)
  if i % 100 == 0:
    print(i, myloss)

  if i == 3900:
    break 


pos_filenames = ['/home/chaoji/Downloads/aclImdb/test/pos/pos.txt']
neg_filenames = ['/home/chaoji/Downloads/aclImdb/test/neg/neg.txt']

vocab_path = 'vocab'

data = [tf.io.gfile.GFile(fn) for fn in pos_filenames + neg_filenames]
data = itertools.chain(*data)

subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)

logits = []
for l in data:
  token_ids = np.reshape(subtokenizer.encode(l, add_eos=True), [1, -1])
  logits.append(model.call(token_ids, training=False).numpy()[0, 0])



