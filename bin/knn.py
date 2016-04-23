#! /usr/bin/env python

"""
Show the knn words in the embeddings to a given word.

Usage:
   ./knn.py embeddings vocabulary

Options:
  -h, --help              : display this help and exit
"""
## Required
#
# sudo apt-get install build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev libatlas-base-dev libatlas3gf-base
# sudo apt-get remove libopenblas-base
# sudo pip install --upgrade nose
# sudo pip install -U scikit-learn

from __future__ import print_function
import sys
from optparse import OptionParser
from math import sqrt
from operator import itemgetter
import re
# clustering
from scipy.cluster.vq import kmeans, whiten, vq
from sklearn.cluster import dbscan
import numpy as np

# Number of neighbors to return.
top = 10
# min number in cluster
min_core = 3
# Cluster representatives
representatives = 5

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

def knn(embeddings, id_word, word_id):
  """Show closest k words"""
  input = sys.stdin
  while True:
    word = input.readline()
    if not word: break
    word = word.strip().decode('utf-8')
    word = normalize(word, word_id)
    if not word:
      print("OOV word")
      continue
    # numpy version
    i = 0
    for index, distance2 in l2_nearest(embeddings, embeddings[word_id[word]], top+1):
      print('%i\t%s\t%f' % (i, id_word[index].encode('utf-8'), sqrt(distance2)))
      i += 1

def Kmeans(file, vocabfile, k):
  np.random.seed((1000,2000))
  whitened = whiten(embeddings)
  codebook, distortion = kmeans(whitened, k)
  clusters = [l2_nearest(embeddings, c, representatives+1) for c in codebook]
  # output
  print(len(codebook), distortion)
  for centroid in codebook:
    print(' '.join([str(x) for x in centroid]))
  print()
  for cluster in clusters:
    print(' '.join([id_word[i] for i, d in cluster]).encode('utf-8'))
  print()
  # assign clusters to words
  codes, _ = vq(embeddings, codebook)
  for w, c in zip(word_id.keys(), codes):
    print(w, c)

def Dbscan(embeddings, id_word, word_id, eps, min_size):
  coreSamples, labels = dbscan(embeddings, eps, min_size)
  # group clusters
  clusters = {}
  for i, label in enumerate(labels):
    if label not in clusters:
      clusters[label] = []
    clusters[label].append(id_word[i].encode('utf-8'))
  # output
  print(len(clusters) - 1)
  for c in clusters.iterkeys():
    if c < 0: continue          # -1 is noise
    print(' '.join([str(x) for x in embeddings[int(c)]]))
  print()
  # show clusters
  for c, words in clusters.iteritems():
    print(c, ' '.join(words))

def readClusters(clusterfile):
  cfile = open(clusterfile)
  k = cfile.readline().split()[0]
  clusters = []
  for i in range(int(k)):
    vector = [float(x) for x in cfile.readline().split()]
    clusters.append(vector)
  return clusters

def annotate(embeddings, id_word, word_id, clusterfile, col = 0):
  clusters = readClusters(clusterfile)
  for line in sys.stdin:
    line = line.strip().decode('utf-8')
    if not line:
      print()
      continue
    attrs = line.split('\t')
    # detect which column to use
    if not col:
      if attrs[8] == '_':
        col = 8                 # PHEAD
      else:
        col = 9                 # PDEPREL
    # get vector for token
    token = attrs[1]            # form
    token = normalize(token, word_id)
    if not token:
      token = attrs[2]          # try lemma
      token = normalize(token, word_id)
    if token:
      id = word_id[token]
    else:
      id = 0 # word_id['<UNK>']
    e = embeddings[id]
    # find cluster
    min = 1e12
    for i, cluster in enumerate(clusters):
      d2 = ((cluster - e) ** 2).sum()
      if d2 < min:
        min = d2
        c = i
    attrs[col] = 'C%i' % (c)
    print('\t'.join(attrs).encode('utf-8'))

def loadVocab(vocab_file):
  vocab = []
  with open(vocab_file, 'rb') as file:
    for line in file:
      vocab.append(line.strip().decode('utf-8'))
  return vocab

def loadEmbeddings(filename, vocab_file=None):
  vocab = []
  if vocab_file:
    vocab = loadVocab(filename)
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

def main():
  usage = """usage: %prog [options] embeddings [vocabulary]
Show knn of words typed on stdin."""
  parser = OptionParser(usage=usage)
  parser.add_option("-d", "--dbscan",
                    action="store", type="float", default=0.0,
                    help="Create clusters of distance EPS to stdout using dbscan",
                    metavar="EPS")
  parser.add_option("-k", "--kmeans",
                    action="store", type="int", default=0,
                    help="Create N clusters to stdout using kmeans",
                    metavar="N")
  parser.add_option("-a", "--annotate", metavar="FILE",
                    help="Annotate CoNLL-X input with clusters from FILE")
  parser.add_option("-c", "--col",
                    action="store", type="int", default=0,
                    help="Column where to put cluster annotation",
                    metavar="C")
  parser.add_option("-g", "--group", metavar="FILE",
                    help="Show clusters from FILE")
  parser.add_option("-f", "--format", type="string", default="plain",
                    help="Embeddings file format: plain (default), word2vec")
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

  # Map words to indices
  word_id = { v:i for i,v in enumerate(id_word)}

  if options.kmeans:
    Kmeans(embeddings, id_word, word_id, options.kmeans)
  elif options.dbscan:
    Dbscan(embeddings, id_word, word_id, options.dbscan, min_core)
  elif options.group:
    # print(the clusters
    codebook = readClusters(options.group)
    codes, _ = vq(embeddings, np.array(codebook))
    groups = {}
    for i,c in enumerate(codes):
      if c not in groups:
        groups[c] = []
      groups[c].append(id_word[i])
    for c,members in groups.iteritems():
      print(c, ' '.join(members).encode('utf-8'))
  elif options.annotate:
    anontate(embeddings, id_word, word_id, options.annotate, options.col)
  else:
    knn(embeddings, id_word, word_id)

if __name__ == '__main__':
  main()
