#!/usr/bin/python
# -*- coding: utf-8 -*-

import theano
import numpy as np
from theano import tensor as T
from theano.compile.io import In
from collections import OrderedDict

from .utils import RNN

class ElmanBRNN(RNN):

    def __init__(self, num_hidden, num_classes, context_win_size, embeddings,
                 featdim=0, fine_tuning=False, truncate_gradient=-1):
        """
        num_hidden :: dimension of the hidden layer
        num_classes :: number of classes
        context_win_size :: word window context size
        embeddings :: matrix
        """
        # hyper parameters of the model

        self.hyperparams = {}

        # nh :: dimension of the hidden layer
        nh = num_hidden
        self.hyperparams['nh'] = nh

        # nc :: number of classes
        nc = num_classes
        self.hyperparams['nc'] = nc

        # de :: dimension of the word embeddings
        de = embeddings.shape[1]
        self.hyperparams['de'] = de

        # cs :: word window context size
        cs = context_win_size
        self.hyperparams['cs'] = cs

        self.hyperparams['featdim'] = featdim
        self.hyperparams['fine_tuning'] = fine_tuning
        self.hyperparams['truncate_gradient'] = truncate_gradient

        # parameters of the model
        self.emb = theano.shared(embeddings.astype(theano.config.floatX))

        # inputs
        idxs = T.imatrix()
        w = T.fscalar('w')
        x = self.emb[idxs].reshape((idxs.shape[0], de * cs))*w
        y = T.iscalar('y')
        y_sentence = T.ivector('y_sentence')
        f = T.matrix('f')
        f.reshape((idxs.shape[0], featdim))

        # forward parameters of the model
        self.fWx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                         (de * cs, nh)).astype(theano.config.floatX))

        self.fWh = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                         (nh, nh)).astype(theano.config.floatX))

        self.fbh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.fh0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        fparams = [self.fWx, self.fWh, self.fbh, self.fh0]
        fnames = ['fWx', 'fWh', 'fbh', 'fh0']

        def frecurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.fWx) + T.dot(h_tm1, self.fWh) + self.fbh)
            return h_t

        fh, _ = theano.scan(fn=frecurrence,
                            sequences=x,
                            outputs_info=[self.fh0],
                            n_steps=x.shape[0],
                            truncate_gradient=truncate_gradient)

        # backwards parameters of the model
        self.bWx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                         (de * cs, nh)).astype(theano.config.floatX))

        self.bWh = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                         (nh, nh)).astype(theano.config.floatX))

        self.bbh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.bh0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        bparams = [self.bWx, self.bWh, self.bbh, self.bh0]
        bnames = ['bWx', 'bWh', 'bbh', 'bh0']

        def brecurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.bWx) + T.dot(h_tm1, self.bWh) + self.bbh)
            return h_t

        bh, _ = theano.scan(fn=brecurrence,
                            sequences=x,
                            outputs_info=[self.bh0],
                            n_steps=x.shape[0],
                            go_backwards=True,
                            truncate_gradient=truncate_gradient)

        # inverting backwards hidden
        bh = bh[::-1]

        # concatenation parameters
        self.bW = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                        (nh+featdim, nc)).astype(theano.config.floatX))

        self.fW = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                        (nh+featdim, nc)).astype(theano.config.floatX))

        self.b = theano.shared(np.zeros(nc, dtype=theano.config.floatX))

        # adding features
        if featdim > 0:
            fh_final = T.concatenate([fh, f], axis=1)
            bh_final = T.concatenate([bh, f], axis=1)
        else:
            fh_final = fh
            bh_final = bh

        # "concatenating" forward and backward hidden states
        h = T.dot(bh_final, self.bW) + T.dot(fh_final, self.fW)

        s = T.nnet.softmax(h + self.b)

        p_y_given_x_lastword = s[-1, :]
        p_y_given_x_sentence = s

        self.params = fparams + bparams + [self.bW, self.fW, self.b]
        self.names = fnames + bnames + ['bW', 'fW', 'b']

        if fine_tuning:
            self.params.append(self.emb)
            self.names.append("emb")

        # prediction
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost functions
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])

        nll = -T.mean(T.log(p_y_given_x_lastword)[y])

        # gradients
        sentence_gradients = T.grad(sentence_nll, self.params)
        gradients = T.grad(nll, self.params)

        # learning rate
        lr = T.scalar('lr')

        # updates
        sentence_updates = OrderedDict((p, p - lr * g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        updates = OrderedDict((p, p - lr * g)
                              for p, g in
                              zip(self.params, gradients))

        # theano functions
        self.classify = theano.function(inputs=[idxs, f, In(w, value=1.0)],
                                        outputs=y_pred,
                                        on_unused_input='ignore')

        self.sentence_train = theano.function(inputs=[idxs, f, y_sentence, lr, In(w, value=1.0)],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              on_unused_input='ignore')

        self.train = theano.function(inputs=[idxs, f, y, lr, In(w, value=1.0)],
                                     outputs=nll,
                                     updates=updates,
                                     on_unused_input='ignore')

        self.predict = theano.function(inputs=[idxs, f, In(w, value=1.0)],
                                       outputs=p_y_given_x_sentence,
                                       on_unused_input='ignore')

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:\
                                                  self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})
