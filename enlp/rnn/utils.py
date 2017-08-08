#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
From:
https://github.com/mesnilgr/is13/
"""

import random
import json
import numpy as np
import os


class RNN(object):

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())

        if 'emb' not in self.names:
            np.save(os.path.join(folder, 'emb.npy'), self.emb.get_value())

        self.hyperparams["name"] = self.name
        with open(os.path.join(folder, "hyper.json"), "wb") as json_file:
            json.dump(self.hyperparams, json_file)

    @classmethod
    def load(cls, folder):
        """
        Loads by params from folder by name
        Args:
            folder: folder from where to read param values (in npy format)

        Returns:
            None

        """
        with open(os.path.join(folder, "hyper.json"), 'rb') as json_file:
            hyperparams = json.load(json_file)

        assert cls.__name__ == hyperparams["name"]

        embeddings = np.load(os.path.join(folder, 'emb.npy'))

        rnn = cls(num_hidden=hyperparams["nh"],
                  num_classes=hyperparams["nc"],
                  context_win_size=hyperparams["cs"],
                  embeddings=embeddings,
                  featdim=hyperparams["featdim"],
                  fine_tuning=hyperparams["fine_tuning"],
                  truncate_gradient=hyperparams["truncate_gradient"])

        folder_param_names = [filename.replace('.npy', '') for filename in os.listdir(folder)
                              if "emb" not in filename]

        for param, name in zip(rnn.params, rnn.names):
            if name in folder_param_names:
                param.set_value(np.load(os.path.join(folder, name + '.npy')))
                print "loaded " + name

        return rnn


def shuffle(lol, seed):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def minibatch(l, bs):
    """
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    """
    out = [l[:i]for i in xrange(1, min(bs, len(l)+1))]
    out += [l[i-bs:i] for i in xrange(bs, len(l)+1)]
    assert len(l) == len(out)
    return out


def contextwin(l, win):
    """
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence

    OBS:
        assumes PADDING index is -1

    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [lpadded[i:i+win] for i in range(len(l))]

    assert len(out) == len(l)
    return out
