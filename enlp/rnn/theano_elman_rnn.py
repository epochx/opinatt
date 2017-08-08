#!/usr/bin/python
# -*- coding: utf-8 -*-


import theano
import numpy as np
from theano import tensor as T
from theano.compile.io import In
from collections import OrderedDict

from .utils import RNN


class ElmanRNN(RNN):

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
        self.Wx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                        (de * cs, nh)).astype(theano.config.floatX))

        self.Wh = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                        (nh, nh)).astype(theano.config.floatX))

        self.W = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,
                                                       (nh+featdim, nc)).astype(theano.config.floatX))

        self.bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.b = theano.shared(np.zeros(nc, dtype=theano.config.floatX))

        self.h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.emb = theano.shared(embeddings.astype(theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        if fine_tuning:
            self.params.append(self.emb)
            self.names.append("emb")

        idxs = T.imatrix()
        w = T.fscalar('w')
        x = self.emb[idxs].reshape((idxs.shape[0], de * cs))*w
        y = T.iscalar('y')
        y_sentence = T.ivector('y_sentence')
        f = T.matrix('f')
        f.reshape((idxs.shape[0], featdim))

        def recurrence(x_t, feat_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)

            if featdim > 0:
                all_t = T.concatenate([h_t, feat_t])
            else:
                all_t = h_t

            s_t = T.nnet.softmax(T.dot(all_t, self.W) + self.b)

            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=[x, f],
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0],
                                truncate_gradient=truncate_gradient)

        # probabilities s.shape = [steps, 1, nclasses]
        p_y_given_x_lastword = s[-1, 0, :]
        p_y_given_x_sentence = s[:, 0, :]

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
        self.classify = theano.function(inputs=[idxs, f, In(w, value=1.0)], outputs=y_pred)

        self.sentence_train = theano.function(inputs=[idxs, f, y_sentence, lr, In(w, value=1.0)],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

        self.train = theano.function(inputs=[idxs, f, y, lr, In(w, value=1.0)],
                                     outputs=nll,
                                     updates=updates)

        self.predict = theano.function(inputs=[idxs, f, In(w, value=1.0)],
                                       outputs=p_y_given_x_sentence)

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

