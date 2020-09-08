import tensorflow as tf
import glob
import numpy as np
from model import SequenceClassifier
import itertools
from commons import tokenization


def _parse_example(serialized_example):

  parse_dict = {'token_ids': tf.io.VarLenFeature(tf.int64),
                'label': tf.io.VarLenFeature(tf.int64)}

  parsed = tf.io.parse_single_example(serialized_example, parse_dict)
  token_ids = tf.sparse.to_dense(parsed['token_ids'])
  label = tf.sparse.to_dense(parsed['label'])

  return token_ids, label

_READ_RECORD_BUFFER = 8 * 1000 * 1000
shuffle = True
num_parallel_calls = 8
random_seed = 0
filenames = sorted(glob.glob('labeled/*.tfrecord'))


from dataset import DatasetBuilder

bucket_width = 100
num_buckets = 10
batch_size = 64 
shuffle = True
num_parallel_calls = 8
random_seed = 42 


builder = DatasetBuilder(batch_size=batch_size,
                           shuffle=shuffle,
                           max_length=None,
                           num_parallel_calls=num_parallel_calls,
                           num_buckets=num_buckets,
                           bucket_width=bucket_width,
                           random_seed=random_seed)

dataset = builder.build_finetune_dataset(filenames)


vocab_path = 'vocab'
subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)

vocab_size = subtokenizer.vocab_size

hidden_size = 512 

model = SequenceClassifier(vocab_size, hidden_size, dropout=0.2, dropout_embedding=True) 

ckpt_path = 'lm' #'sa'
ckpt = tf.train.Checkpoint(model=model)
latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
print('Loaded latest checkpoint ', latest_ckpt)
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
  myloss = train_step(token_ids, labels)
  if i % 100 == 0:
    print(i, myloss)

  if i == 1200:
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



