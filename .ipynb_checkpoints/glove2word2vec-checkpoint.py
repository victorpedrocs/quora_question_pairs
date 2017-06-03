#!/usr/bin/python3
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec('datasets/glove.6B.300d.txt', 'datasets/glove.6B.300d.word2vec')
