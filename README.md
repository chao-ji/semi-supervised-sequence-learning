# Semi-supervised-sequence-learning

This repo documents the experiments that reproduce the results presented in the paper [semi-supervised-sequence-learning](https://arxiv.org/abs/1511.01432). Briefly, we pretrain a Sequence Autoencoder or Language Model on *unlabeled* text data, and then fine-tune an RNN-based sequence classifier initialized with the pretrained weights using *labeled* text data, which gives rises to better classification accuracy than weights initialized randomly. 

## Data Preparation

### IMDB dataset

We use [IMDB movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) for this experiment. After downloading and untarring the [file](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), navigate to the directory `aclImdb/train` which holds the positive (`aclImdb/train/pos`) labeled and negative (`aclImdb/train/neg`) labeled reviews in the training split, as well as unlabeled reviews (`aclImdb/train/unsup`). Then `cd` into each of the subdirectories and run

```
for f in *.txt; do (cat "${f}"; echo) >> pos.txt; done
```
 
```
for f in *.txt; do (cat "${f}"; echo) >> neg.txt; done
```

```
for f in *.txt; do (cat "${f}"; echo) >> unlabeled.txt; done
```
in order to concatenate multiple txt files into a single file. Similarly, we run these steps for the test split (`aclImdb/test/pos` and `aclImdb/test/neg`).

### Tfrecord files

We must convert raw text files (e.g. `pos.txt` and `neg.txt`) to `*.tfrecord` files that can be processed by the training pipeline.

```
python run_tfrecord_generation.py \
  --pos_filenames=aclImdb/train/pos/pos.txt \
  --neg_filenames=aclImdb/train/neg/neg.txt \
  --unlabeled_filenames=aclImdb/train/unsup/unlabeled.txt 
```

The resulting `*.tfrecord` files in `labeled` directory stores examples of sequence-label pairs(`aclImdb/train/pos/pos.txt` and `aclImdb/train/neg/neg.txt`) , while those in `unlabeled` directory stores examples of all sequences without labels (`pos.txt`, `neg.txt` and `unlabeled`). Note that the vocabulary file `vocab.subtokens` will be generated along with `*.tfrecord` files, which is needed to tokenize raw text in later stages.

## Pretraining

The pretraining stage involves training a Sequence Autoencoder or a Language Model.

```
python run_pretraining.py \
  --data_dir=unlabeled \
  --vocab_path=vocab
```

`data_dir` is the path to the directory storing unlabeled sequences, and `vocab_path` is the path to the prefix of the vocabulary file.

The pretrained weights will be saved to checkpoint files `sa` (for sequence autoencoder) or `lm` (for language model).


## Fine-tuning

To fine-tune a pre-trained language model (`method=lm`; set `method=sa` for sequence auto-encoder) on labeled dataset, run
```
python run_finetune.py \
  --data_dir=labeled \
  --vocab_path=vocab \
  --method=sa \
  --test_pos_filename=aclImdb/test/pos/pos.txt \
  --test_neg_filename=aclImdb/test/neg/neg.txt
```

Depending on which method was used for the pretraining step, weights (of a single-layer LSTM and an embedding matrix) will be loaded from checkpoints in either `sa` or `lm`, and the LSTM-based sequence classifier will be fine-tuned for dramatically fewer iterations compared to the pretraining stage. 

As the baseline that the fine-tuning approach is compared against, one can train the LSTM-based sequence classifier on labeled data from scatch (i.w. without initializing from pretrained weights):

```
python run_finetune.py \
  --data_dir=labeled \
  --vocab_path=vocab \
  --no_finetune=True \
  --test_pos_filename=aclImdb/test/pos/pos.txt \
  --test_neg_filename=aclImdb/test/neg/neg.txt
```


## Results

The pretraining using either sequence autoencoder or language model improves the classification accuracy dramatically (~92% vs. ~85%).


