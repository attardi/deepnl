#! /usr/bin/env python

"""
Check if phrase is a MWE.

Usage:
   ./mwe.py embeddings vocabulary

Options:
  -h, --help              : display this help and exit
"""

from __future__ import print_function
import sys
from optparse import OptionParser
import cPickle as pickle
from operator import itemgetter
import re
import numpy as np

# Number of neighbors to return.
top = 5

# Normalize digits by replacing them with #
DIGITS = re.compile("[0-9]", re.UNICODE)

def case_normalizer(word, dictionary):
  """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedures.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
  w = word
  lower = (dictionary.get(w.lower(), 1e12), w.lower())
  upper = (dictionary.get(w.upper(), 1e12), w.upper())
  title = (dictionary.get(w.title(), 1e12), w.title())
  results = [lower, upper, title]
  results.sort()
  index, w = results[0]
  if index != 1e12:
    return w
  return word

def normalize(word, word_id):
  """ Find the closest alternative in case the word is OOV."""
  if not word in word_id:
      word = DIGITS.sub("0", word)
  if not word in word_id:
      word = case_normalizer(word, word_id)

  if not word in word_id:
      return None
  return word

def l2_nearest(embeddings, e, k):
  """Sort vectors according to their Euclidean distance from e
  and return the k closest.
  Returns list of (index, distance^2)
  """

  distances = ((embeddings - e) ** 2).sum(axis=1) # ** 0.5
  sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
  return sorted_distances[1:k]

def variant(word, id, embeddings):
  # FIXME: should use POS
  if len(word) > 3:
    return l2_nearest(embeddings, embeddings[id], top+1)
  else:
    return [(id,0)]

def closest(ngram, word_id, id_word, embeddings):
    for i,word in enumerate(ngram):
      for index, distance2 in variant(word, word_id[word], embeddings):
        yield [w if n!=i else id_word[index] for n, w in enumerate(ngram)]

def show(embeddings, word_id, id_word, counts):
  """Show closest k phrases"""
  input = sys.stdin
  while True:
    words = input.readline()
    if not words: break
    words = words.strip().decode('utf-8').split()
    words = [normalize(word, word_id) for word in words]
    if not all(words):
      print("OOV word")
      continue
    phrase = ' '.join(words)
    freq = counts.get(phrase, 0)
    print(phrase.encode('utf-8'), freq)
    for ngram in closest(words, word_id, id_word, embeddings):
      phrase = ' '.join(ngram)
      freq = counts.get(phrase, 0)
      print(phrase.encode('utf-8'), freq)

def loadVocab(vocab_file):
  vocab = []
  with open(vocab_file, 'rb') as file:
    for line in file:
      vocab.append(line.strip().decode('utf-8'))
  return vocab

def loadEmbeddings(filename, vocab_file=None):
  vocab = []
  if vocab_file:
    vocab = loadVocab(vocab_file)
    with open(filename, 'rb') as file:
      vectors = np.array([[float(value) for value in line.split()]
                          for line in file])
  else:
    # read both from same file in word2vec format
    vectors = []
    with open(filename, 'rb') as file:
      len, size = file.readline().strip().split()
      for line in file:
        items = line.split()
        vocab.append(items[0].decode('utf-8'))
        vectors.append([float(value) for value in items[1:]])
      vectors = np.array(vectors)

  return vectors, vocab

def PolyglotLoad(filename):
  """
  Load the feature matrix used by word2embeddings.
  """
  vectors = []
  with open(filename, 'rb') as f:
    for line in f:
      items = line.split()
      word = unicode(items[0], 'utf-8')
      vectors.append([float(x) for x in items[1:]])
  return np.array(vectors)

def main():
  usage = """usage: %prog [options] embeddings [vocabulary]
Show knn of variant of phrase typed on stdin."""
  parser = OptionParser(usage=usage)
  parser.add_option("-f", "--format", type="string", default="plain",
                    help="Embedding file format: plain (default), word2vec")
  parser.add_option("-c", "--counts", type="string",
                    help="Ngram frequencysfile")
  options, args = parser.parse_args()
  if len(args) == 0:
    parser.error("incorrect number of arguments")

  file = args[0]
  if len(args) == 2:
    vocab_file = args[1]
  else:
    vocab_file = None

  if options.format.lower() == 'word2vec':
    embeddings, id_word = loadEmbeddings(file)
  elif options.format.lower() == 'word2embeddings':
    embeddings = np.load(file).get_word_embeddings()
    id_word = ['<UNK>', '<S>', '</S>', '<PAD>']
    id_word.extend(loadVocab(vocab_file))
  else:
    embeddings, id_word = loadEmbeddings(file, vocab_file)

  counts = {}
  if options.counts:
    with open(options.counts) as file:
      for line in file:
        ngram, freq = line.strip().split()
        ngram = re.sub('_', ' ', ngram)
        counts[ngram] = int(freq)

  # Map words to indices
  word_id = { v:i for i,v in enumerate(id_word)}

  show(embeddings, word_id, id_word, counts)

if __name__ == '__main__':
  main()
