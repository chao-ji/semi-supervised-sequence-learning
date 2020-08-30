"""Defines models for pretraining and fine-tuning."""
import tensorflow as tf

from commons.layers import EmbeddingLayer


class Pretrainer(tf.keras.Model):
  """Base class of pretraining methods."""
  def __init__(self, vocab_size, hidden_size, dropout_rate=0.2):
    """Constructor.

    Args:
      vocab_size: int scalar, num of subword tokens (including SOS/PAD and EOS) 
        in the vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(Pretrainer, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate

    self._embedding_logits_layer = EmbeddingLayer(
        vocab_size, hidden_size, scale_embeddings=False)
    self._recurrent_layer = tf.keras.layers.LSTM(hidden_size,
                                                 return_sequences=True,
                                                 return_state=True,
                                                 dropout=dropout_rate)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)


class SequenceAutoencoder(Pretrainer):
  """Sequence Autoencoder that encodes a sequence into vector representation and
  decodes it back to the original sequence. 
  """
  def call(self, token_ids, training=True):
    """Computes the logits of each word in the decoded sequence given the 
    original sequence to be encoded.

    Args:
      token_ids: int tensor of shape [batch_size, seq_len], token ids
        of the sequence to be encoded.
      training: bool scalar, True if in training mode.

    Returns:
      logits: float tensor of shape [batch_size, seq_len - 1, vocab_size].
    """
    src_token_embeddings = self._embedding_logits_layer(
        token_ids, 'embedding')
    tgt_token_embeddings = self._embedding_logits_layer(
        token_ids[:, :-1], 'embedding')
    padding_mask = token_ids != 0

    src_token_embeddings = self._dropout_layer(
        src_token_embeddings, training=training)
    tgt_token_embeddings = self._dropout_layer(
        tgt_token_embeddings, training=training)

    _, h, c = self._recurrent_layer(
        src_token_embeddings, mask=padding_mask, training=training)
    outputs, _, _ = self._recurrent_layer(
        tgt_token_embeddings, initial_state=(h, c), training=training)
    logits = self._embedding_logits_layer(outputs, 'logits')
    return logits


class LanguageModel(Pretrainer):
  """Recurrent language model that predicts what comes next in a sequence"""
  def call(self, token_ids, states, training=True):
    """Computes the logits of each "next word" given the input sequence. 

    Args:
      token_ids: int tensor of shape [batch_size, seq_len], token ids
        of the input sequence.
      states: a tuple of float tensors of shape [batch_size, hidden_size], the 
        `h` and `c` state of LSTM cells.
      training: bool scalar, True if in training mode.

    Returns:
      logits: float tensor of shape [batch_size, seq_len, vocab_size].
      new_states: a tuple of float tensors of shape [batch_size, hidden_size], 
        the updated `h` and `c` state of LSTM cells.
    """
    token_embeddings = self._embedding_logits_layer(
        token_ids, 'embedding')
    token_embeddings = self._dropout_layer(token_embeddings, training=training)
    outputs, h, c = self._recurrent_layer(
        token_embeddings, states, training=training)

    logits = self._embedding_logits_layer(outputs, 'logits')
    return logits, (h, c)


class SequenceClassifier(tf.keras.Model):
  """Recurrent network that classifies sequences as positive or negative."""
  def __init__(self, vocab_size, hidden_size, dropout=0.0):
    """Constructor.

    Args:
      vocab_size: int scalar, num of subword tokens (including SOS/PAD and EOS) 
        in the vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(SequenceClassifier, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._dropout = dropout

    self._embedding_logits_layer = EmbeddingLayer(
        vocab_size, hidden_size, scale_embeddings=False)
    self._recurrent_layer = tf.keras.layers.LSTM(
        self._hidden_size, dropout=dropout)

    self._dense_layer = tf.keras.layers.Dense(hidden_size, activation='relu')
    self._logits_layer = tf.keras.layers.Dense(1)

  def call(self, token_ids, training=False):
    """Computes the logits of the classes of the input sequences.

    Args:
      token_ids: int tensor of shape [batch_size, seq_len], token ids
        of the input sequence.
      training: bool scalar, True if in training mode.

    Returns:
      logits: float tensor of shape [batch_size, 1]
    """
    padding_mask = tf.equal(token_ids, 0)
    embeddings = self._embedding_logits_layer(token_ids, 'embedding')
    logits = self._logits_layer(self._dense_layer(self._recurrent_layer(
        embeddings, mask=tf.logical_not(padding_mask), training=training)))

    return logits
