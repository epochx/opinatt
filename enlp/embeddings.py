#!/usr/bin/python
# -*- coding: utf-8 -*-

# based on https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py

import os
from .settings import DATA_PATH
import numpy as np
import warnings

EMBEDDINGS_PATH = os.path.join(DATA_PATH, 'word_embeddings')


def make_closing(base, **attrs):
    """
    Add support for `with Base(attrs) as fout:` to the base class if it's 
    missing. The base class' `close()` method will be called on context exit,
    to always close the file properly.

    This is needed for gzip.GzipFile, bz2.BZ2File etc in older Pythons (<=2.6),
    which otherwise raise "AttributeError: GzipFile instance has no attribute
    '__exit__'".
    """
    if not hasattr(base, '__enter__'):
        attrs['__enter__'] = lambda self: self
    if not hasattr(base, '__exit__'):
        attrs['__exit__'] = lambda self, type, value, traceback: self.close()
    return type('Closing' + base.__name__, (base, object), attrs)


def smart_open(fname, mode='rb'):
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return make_closing(BZ2File)(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return make_closing(GzipFile)(fname, mode)
    return open(fname, mode)


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).
    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key])
                for key in sorted(self.__dict__)
                if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Embeddings(object):

    filepath = None
    unseen = None
    binary = None

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self, filepath=None, fvocab=None, binary=False,
                 encoding='utf8', unicode_errors='strict',
                 limit=None, datatype=np.float32, unseen='mikolov'):

        def add_word(word, weights):
                word_id = len(self.vocab)
                if word in self.vocab:
                    # logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    # most common scenario: no vocab file given
                    # just make up some bogus counts, in descending order
                    self.vocab[word] = Vocab(index=word_id,
                                             count=vocab_size-word_id)
                elif word in counts:
                    # use count from the vocab file
                    self.vocab[word] = Vocab(index=word_id,
                                             count=counts[word])
                else:
                    # vocab file given, but word is missing -- set count to None
                    # logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    self.vocab[word] = Vocab(index=word_id, count=None)
                self.syn0[word_id] = weights
                self.index2word.append(word)

        if self.binary:
            binary = self.binary

        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.unseen_vocab = {}

        counts = None
        if fvocab is not None:
            counts = {}
            with open(fvocab) as fin:
                for line in fin:
                    word, count = unicode(line).strip().split()
                    counts[word] = int(count)

        if filepath is None and self.filepath:
            filepath = self.filepath

        # logger.info("loading projection weights from %s", fname)
        with smart_open(filepath) as fin:
            lines = fin.readlines()
            header = unicode(lines[0], encoding)

            # throws for invalid file format
            try:
                vocab_size, vector_size = map(int, header.split())
            except ValueError:
                vocab_size = len(lines)
                vector_size = len(header.split())-1
                binary = False

            if limit:
                vocab_size = min(vocab_size, limit)

            self.vector_size = int(vector_size)
            self.layer1_size = int(vector_size)
            self.syn0 = np.zeros((vocab_size, vector_size), dtype=datatype)

        with smart_open(filepath) as fin:
            if binary:
                # skip first line *header*
                fin.readline()
                binary_len = np.dtype(np.float32).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("""
                                           Unexpected end of input; is count
                                           incorrect or file otherwise damaged?
                                           """)
                        # ignore newlines in front of words (some binary files have)
                        if ch != b'\n':
                            word.append(ch)
                    word = unicode(b''.join(word), encoding, errors=unicode_errors)
                    weights = np.fromstring(fin.read(binary_len),
                                            dtype=np.float32)
                    add_word(word, weights)
            else:
                for line_no in xrange(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError("""
                                       Unexpected end of input; is count
                                       incorrect or file otherwise damaged?
                                       """)
                    parts = unicode(line.rstrip(), encoding,
                                    errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("""
                                         invalid vector on line %s
                                          (is this really the text format?)
                                         """ % (line_no))
                    word, weights = parts[0], list(map(np.float32, parts[1:]))
                    add_word(word, weights)
        if self.syn0.shape[0] != len(self.vocab):
            # logger.info(
            #    "duplicate words detected, shrinking matrix size from %i to %i",
            #    result.syn0.shape[0], len(result.vocab)
            # )
            self.syn0 = np.ascontiguousarray(self.syn0[:len(self.vocab)])
        assert (len(self.vocab), self.vector_size) == self.syn0.shape
        # logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))

        if self.unseen:
            unseen = self.unseen

        if unseen:
            if unseen == 'mikolov':
                self.unseen = self._gen_unseen_mikolov()
            elif unseen == 'kim':
                self.unseen = self._gen_unseen_kim()
            else:
                self.unseen = self._gen_unseen_string(unseen)

    def __getitem__(self, words):
        if isinstance(words, basestring):
            # allow calls like trained_model['office']
            # as a shorthand for trained_model[['office']]
            word = words
            if word in self:
                return self.syn0[self.vocab[word].index]
            else:
                warnings.warn("Returning unseen word")
                return self.unseen()
        elif isinstance(words, list):
            return np.vstack([self[word] for word in words])
        else:
            raise TypeError("Use list or string")

    def __contains__(self, word):
        return word in self.vocab

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (self.__class__.__name__,
                                                    len(self.index2word),
                                                    self.vector_size,
                                                    self.alpha)

    def _gen_unseen_mikolov(self, rate=0.01):
        """
        Based on an answer by Mikolov itself: you can initialize the vector
        based on the space described by the infrequent words.
        In his answer he mentions that you should average the
        infrequent words and in that way build the unknown token.

        Here we uniformly sample a sphere centered in the centroid of low
        frequency words which radius is the mean distance between the centroid
        and all the words in the low frequency set.
        rate
        Based on:
        http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        """
        low_freq_list, low_freq_count = zip(*self._get_low_frequency_words(rate))
        low_freq_matrix = np.vstack([self[word] for word in low_freq_list])
        vector_size = self.vector_size

        # aproximating the centroid of the low frequency terms
        center = np.mean(low_freq_matrix, axis=0)

        # getting the average distance of the centroid
        # to low frequency terms (radius of the n-shpere)
        radius = np.mean([dist(center, self[word])
                          for word in low_freq_list])

        # generate a random vect
        random_vec = np.random.rand(vector_size, )

        # normalizing the random vect
        random_vec /= np.linalg.norm(random_vec)

        # generate the output
        unseen_vector = (radius * (np.random.uniform(0, 1) ** (1 / vector_size)) * random_vec) + center

        def unseen():
            return unseen_vector

        return unseen

    def _gen_unseen_kim(self):
        """
        For words that occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same
        variance as pre-trained ones (this value is for GoogleNews dataset)
        From https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        vector_size = self.vector_size
        unseen_vector = np.random.uniform(-0.25, 0.25, (vector_size,))

        def unseen():
            return unseen_vector

        return unseen

    def _get_low_frequency_words(self, rate=0.001):
        """
        Vocab is a  dict where words are keys and values
        are frequency counts
        """
        counts = [(key, value.count)
                  for key, value in self.vocab.items()]
        sorted(counts, key=lambda x: x[1], reverse=True)
        index = int(len(counts)*rate)
        return counts[-index:]

    def _gen_unseen_string(self, unseen_word):
        def unseen():
            return self[unseen_word]
        return unseen


def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

GoogleNews = type("GoogleNews",
                  (Embeddings,),
                  {"binary": True,
                   "filepath": os.path.join(EMBEDDINGS_PATH,
                                            'GoogleNews-vectors-negative300.bin.gz')})

GoogleFreebase = type("GoogleFreebase",
                      (Embeddings,),
                      {"binary": True,
                       "filepath": os.path.join(EMBEDDINGS_PATH,
                                                'freebase-vectors-skipgram1000.bin.gz')})

WikiDeps = type("WikiDeps",
                (Embeddings,),
                {"binary": False,
                 "filepath": os.path.join(EMBEDDINGS_PATH,
                                          'deps.words.bz2')})

SennaEmbeddings = type("SennaEmbeddings",
                       (Embeddings,),
                       {"binary": False,
                        "unseen": 'UNKNOWN',
                        "filepath": os.path.join(EMBEDDINGS_PATH,
                                                 'senna_embeddings.txt')})

# more embeddings at:
# https://www.quora.com/Where-can-I-find-some-pre-trained-word-vectors-for-natural-language-processing-understanding
# https://github.com/3Top/word2vec-api

# collobert? http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-original.EMBEDDING_SIZE=25.txt.gz

