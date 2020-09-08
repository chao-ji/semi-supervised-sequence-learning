"""Script that parses and tokenizes the raw text files from the IMDB dataset to 
build the vocabulary. The raw text files are encoded into subword token ids and 
serialized into *.tfrecord files.
"""
import itertools
import os

import tensorflow as tf
from absl import app
from absl import flags

from commons import tokenization
from commons.utils import dict_to_example


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'pos_filenames', None, 'Paths to the text files storing positive examples '
        '(one review per line).')
flags.DEFINE_list(
    'neg_filenames', None, 'Paths to the text files storing negative examples '
        '(one review per line).')
flags.DEFINE_list(
    'unlabeled_filenames', None, 'Paths to the text files storing unlabeled '
        'examples (one review per line).')
flags.DEFINE_float(
    'file_byte_limit', 1e6, 'Number of bytes to read from each text file.')
flags.DEFINE_integer(
    'target_vocab_size', 32768, 'The desired vocabulary size. Ignored if '
        '`min_count` is not None.')
flags.DEFINE_integer(
    'threshold', 327, 'If the difference between actual vocab size and '
        '`target_vocab_size` is smaller than this, the binary search '
        'terminates. Ignored if `min_count` is not None.')
flags.DEFINE_integer(
    'min_count', 6, 'The minimum count required for a subtoken to be '
        'included in the vocabulary.')
flags.DEFINE_string(
    'vocab_name', 'vocab', 'Vocabulary will be stored in two files: '
        '"vocab.subtokens", "vocab.alphabet".')
flags.DEFINE_integer(
    'total_shards', 100, 'Total number of shards of the dataset (number of the '
        'generated TFRecord files)')


def main(_):
  pos_filenames = FLAGS.pos_filenames
  neg_filenames = FLAGS.neg_filenames
  unlabeled_filenames = FLAGS.unlabeled_filenames
  file_byte_limit = FLAGS.file_byte_limit
  target_vocab_size = FLAGS.target_vocab_size
  threshold = FLAGS.threshold
  min_count = FLAGS.min_count
  vocab_name = FLAGS.vocab_name
  total_shards = FLAGS.total_shards

  subtokenizer = tokenization.create_subtokenizer_from_raw_text_files(
      pos_filenames + neg_filenames + unlabeled_filenames,
      target_vocab_size,
      threshold,
      min_count=min_count,
      file_byte_limit=file_byte_limit)
  subtokenizer.save_to_file(vocab_name)

  # labeled data for fine-tuning 
  data = [tf.io.gfile.GFile(fn) for fn in pos_filenames + neg_filenames]
  data = itertools.chain(*data)
  labels = [1] * 12500 + [0] * 12500
  generate_tfrecords(subtokenizer, data, total_shards, 'labeled', labels)
  
  # unlabeled data for pretraining
  data = [tf.io.gfile.GFile(fn) for fn in 
      pos_filenames + neg_filenames + unlabeled_filenames]
  data = itertools.chain(*data)
  generate_tfrecords(subtokenizer, data, total_shards, 'unlabeled')


def generate_tfrecords(subtokenizer, data, total_shards, output_dir, labels=None):
  """Generates TFRecord files for labeled or unlabeled datasets.

  Args:
    subtokenizer: a SubTokenizer instance.
    data: an iterable of strings, the stream of text to be tokenized.
    total_shards: int scalar, total number of *.tfrecord files.
    output_dir: string scalar, directory to which *.tfrecord files will be 
      written to.
    labels: (Optional), the list of integers indicating the labels (1 for 
      positives and 0 for negatives).
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  labels = itertools.repeat(None) if labels is None else labels

  filepaths = [os.path.join(output_dir, '%05d-of-%05d.tfrecord' %
      (i + 1, total_shards)) for i in range(total_shards)]

  writers = [tf.io.TFRecordWriter(fn) for fn in filepaths]
  shard = 0

  for line, label in zip(data, labels):
    line = line.strip()

    l = subtokenizer.encode(line, add_eos=True)

    if label is not None:
      example_dict = {'token_ids': l, 'label': [label]}
    else:
      example_dict = {'token_ids': l}

    example = dict_to_example(example_dict)

    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % total_shards

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  flags.mark_flag_as_required('pos_filenames')
  flags.mark_flag_as_required('neg_filenames')
  flags.mark_flag_as_required('unlabeled_filenames')

  app.run(main)
