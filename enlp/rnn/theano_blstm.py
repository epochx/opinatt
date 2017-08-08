#!/usr/bin/python
# -*- coding: utf-8 -*-

import theano
import numpy as np
from theano import tensor as T
from theano.compile.io import In
from collections import OrderedDict

from .utils import RNN

dtype = theano.config.floatX
uniform = np.random.uniform
sigma = T.nnet.sigmoid
softmax = T.nnet.softmax


class BLSTM(RNN):

    def __init__(self, num_hidden, num_classes, context_win_size, embeddings,
                 featdim=0, fine_tuning=False, truncate_gradient=-1):
        """
        num_hidden :: dimension of the hidden layer
        num_classes :: number of classes
        embeddings :: matrix
        featdim :: size of the features
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

        # add one for PADDING at the end
        self.emb = theano.shared(embeddings.astype(theano.config.floatX))

        n_in = de * cs
        n_hidden = n_i = n_c = n_o = n_f = nh
        n_y = nc

        idxs = T.imatrix()
        w = T.fscalar('w')
        # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de * cs))*w
        f = T.matrix('f')
        f.reshape((idxs.shape[0], featdim))
        y = T.iscalar('y')  # label
        y_sentence = T.ivector('y_sentence')

        # forward weights

        self.fW_xi = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_i)).astype(dtype))

        self.fW_hi = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_i)).astype(dtype))

        self.fW_ci = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_i)).astype(dtype))

        self.fb_i = theano.shared(np.cast[dtype](uniform(-0.5, .5, size=n_i)))

        self.fW_xf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_f)).astype(dtype))

        self.fW_hf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_f)).astype(dtype))

        self.fW_cf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_f)).astype(dtype))

        self.fb_f = theano.shared(np.cast[dtype](uniform(0, 1., size=n_f)))

        self.fW_xc = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_c)).astype(dtype))

        self.fW_hc = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_c)).astype(dtype))

        self.fb_c = theano.shared(np.zeros(n_c, dtype=dtype))

        self.fW_xo = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_o)).astype(dtype))

        self.fW_ho = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_o)).astype(dtype))

        self.fW_co = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_o)).astype(dtype))

        self.fb_o = theano.shared(np.cast[dtype](uniform(-0.5, .5, size=n_o)))

        self.fc0 = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.fh0 = T.tanh(self.fc0)

        fparams = [self.fW_xi, self.fW_hi, self.fW_ci, self.fb_i, self.fW_xf,
                   self.fW_hf, self.fW_cf, self.fb_f, self.fW_xc, self.fW_hc,
                   self.fb_c, self.fW_xo, self.fW_ho, self.fW_co, self.fb_o, self.fc0]

        fnames = ['fW_xi', 'fW_hi', 'fW_ci', 'fb_i', 'fW_xf',
                  'fW_hf', 'fW_cf', 'fb_f', 'fW_xc', 'fW_hc',
                  'fb_c', 'fW_xo', 'fW_ho', 'fW_co', 'fb_o', 'fc0']

        def frecurrence(x_t, h_tm1, c_tm1):
            i_t = sigma(theano.dot(x_t, self.fW_xi)
                        + theano.dot(h_tm1, self.fW_hi)
                        + theano.dot(c_tm1, self.fW_ci)
                        + self.fb_i)

            f_t = sigma(theano.dot(x_t, self.fW_xf)
                        + theano.dot(h_tm1, self.fW_hf)
                        + theano.dot(c_tm1, self.fW_cf)
                        + self.fb_f)

            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, self.fW_xc)
                                             + theano.dot(h_tm1, self.fW_hc)
                                             + self.fb_c)

            o_t = sigma(theano.dot(x_t, self.fW_xo)
                        + theano.dot(h_tm1, self.fW_ho)
                        + theano.dot(c_t, self.fW_co)
                        + self.fb_o)

            h_t = o_t * T.tanh(c_t)

            return [h_t, c_t]

        [fh, _, ], _ = theano.scan(fn=frecurrence,
                                   sequences=[x],
                                   outputs_info=[self.fh0, self.fc0],
                                   n_steps=x.shape[0],
                                   truncate_gradient=truncate_gradient)

        # backward weights

        self.bW_xi = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_i)).astype(dtype))

        self.bW_hi = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_i)).astype(dtype))

        self.bW_ci = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_i)).astype(dtype))

        self.bb_i = theano.shared(np.cast[dtype](uniform(-0.5, .5, size=n_i)))

        self.bW_xf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_f)).astype(dtype))

        self.bW_hf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_f)).astype(dtype))

        self.bW_cf = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_f)).astype(dtype))

        self.bb_f = theano.shared(np.cast[dtype](uniform(0, 1., size=n_f)))

        self.bW_xc = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_c)).astype(dtype))

        self.bW_hc = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_c)).astype(dtype))

        self.bb_c = theano.shared(np.zeros(n_c, dtype=dtype))

        self.bW_xo = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_in, n_o)).astype(dtype))

        self.bW_ho = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_hidden, n_o)).astype(dtype))

        self.bW_co = theano.shared(0.2 * uniform(-1.0, 1.0,
                                                 (n_c, n_o)).astype(dtype))

        self.bb_o = theano.shared(np.cast[dtype](uniform(-0.5, .5, size=n_o)))

        self.bc0 = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.bh0 = T.tanh(self.bc0)

        bparams = [self.bW_xi, self.bW_hi, self.bW_ci, self.bb_i, self.bW_xf,
                   self.bW_hf, self.bW_cf, self.bb_f, self.bW_xc, self.bW_hc,
                   self.bb_c, self.bW_xo, self.bW_ho, self.bW_co, self.bb_o, self.bc0]

        bnames = ['bW_xi', 'bW_hi', 'bW_ci', 'bb_i', 'bW_xf',
                  'bW_hf', 'bW_cf', 'bb_f', 'bW_xc', 'bW_hc',
                  'bb_c', 'bW_xo', 'bW_ho', 'bW_co', 'bb_o', 'bc0']

        def brecurrence(x_t, h_tm1, c_tm1):
            i_t = sigma(theano.dot(x_t, self.bW_xi)
                        + theano.dot(h_tm1, self.bW_hi)
                        + theano.dot(c_tm1, self.bW_ci)
                        + self.bb_i)

            f_t = sigma(theano.dot(x_t, self.bW_xf)
                        + theano.dot(h_tm1, self.bW_hf)
                        + theano.dot(c_tm1, self.bW_cf)
                        + self.bb_f)

            c_t = f_t * c_tm1 + i_t * T.tanh(theano.dot(x_t, self.bW_xc)
                                             + theano.dot(h_tm1, self.bW_hc)
                                             + self.bb_c)

            o_t = sigma(theano.dot(x_t, self.bW_xo)
                        + theano.dot(h_tm1, self.bW_ho)
                        + theano.dot(c_t, self.bW_co)
                        + self.bb_o)

            h_t = o_t * T.tanh(c_t)

            return [h_t, c_t]

        [bh, _, ], _ = theano.scan(fn=brecurrence,
                                   sequences=[x],
                                   outputs_info=[self.bh0, self.bc0],
                                   n_steps=x.shape[0],
                                   go_backwards=True,
                                   truncate_gradient=truncate_gradient)

        # concatenation weights

        self.bW = theano.shared(0.2 * uniform(-1.0, 1.0,
                                              (n_hidden + featdim, n_y)).astype(dtype))

        self.fW = theano.shared(0.2 * uniform(-1.0, 1.0,
                                              (n_hidden + featdim, n_y)).astype(dtype))

        self.b = theano.shared(np.zeros(n_y, dtype=dtype))

        # reversing backwards hidden
        bh = bh[::-1]

        # adding features
        if featdim > 0:
            fh_final = T.concatenate([fh, f], axis=1)
            bh_final = T.concatenate([bh, f], axis=1)
        else:
            fh_final = fh
            bh_final = bh

        # "concatenating" hidden states
        h = T.dot(bh_final, self.bW) + T.dot(fh_final, self.fW)

        s = T.nnet.softmax(h + self.b)

        p_y_given_x_lastword = s[-1, :]
        p_y_given_x_sentence = s

        # params and names
        self.params = fparams + bparams + [self.fW, self.bW, self.b]
        self.names = fnames + bnames + ["fW", "bW", "b"]

        if fine_tuning:
            self.params.append(self.emb)
            self.names.append("embeddings")

        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # learning rate
        lr = T.scalar('lr')

        # cost functions
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])

        nll = -T.mean(T.log(p_y_given_x_lastword)[y])

        # gradients
        gradients = T.grad(nll, self.params)
        sentence_gradients = T.grad(sentence_nll, self.params)

        # updates
        updates = OrderedDict((p, p - lr * g)
                              for p, g in zip(self.params, gradients))

        sentence_updates = OrderedDict((p, p - lr * g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions
        self.classify = theano.function(inputs=[idxs, f, In(w, value=1.0)],
                                        outputs=y_pred,
                                        on_unused_input='ignore')

        self.train = theano.function(inputs=[idxs, f, y, lr, In(w, value=1.0)],
                                     outputs=nll,
                                     updates=updates,
                                     on_unused_input='ignore')

        self.sentence_train = theano.function(inputs=[idxs, f, y_sentence, lr, In(w, value=1.0)],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              on_unused_input='ignore')

        self.predict = theano.function(inputs=[idxs, f, In(w, value=1.0)],
                                       outputs=p_y_given_x_sentence,
                                       on_unused_input='ignore')

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')})
