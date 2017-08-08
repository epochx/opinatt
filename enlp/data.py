# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import re
from collections import defaultdict
import random
import warnings
import os

from .settings import PAD, PAD_ID, UNK, UNK_ID


# -------------------- BUILD VOCABULARIES ------------------------------------------------------------------

def start_vocab(add_unk, add_padding):
    vocab = {}
    if add_unk:
        vocab[UNK] = UNK_ID
    if add_padding:
        vocab[PAD] = PAD_ID
    return vocab

def build_token_vocab(sentences, process_token_func, min_freq=1, add_padding=True):
    counts = defaultdict(int)
    sentence_seqs = []
    aspect_words = set()

    for sentence in sentences:
        sentence_seq = []
        for word in sentence.tokens:
            processed_token = process_token_func(word)
            counts[processed_token] += 1
            sentence_seq.append(processed_token)
        sentence_seqs.append(sentence_seq)
        tags_seq, class_ = sent_to_iob_tags(sentence)
        # collect words that are labeled as aspects
        for i, tag in enumerate(tags_seq):
            if tag != "O":
                aspect_words.add(sentence[i].string)

    # get words with low counts but not labeled as aspects
    highcounts = [word for word, count in counts.iteritems()
                  if count >= min_freq or word in aspect_words]

    word2idx = start_vocab(add_unk=True, add_padding=add_padding)
    index = len(word2idx)
    for token in highcounts:
        word2idx[token] = index
        index += 1

    return word2idx

def build_tag_vocab(sentiment=False, add_padding=True):
    if sentiment:
        labels = ["O", "B-TARG:POS", "I-TARG:POS", "B-TARG:NEG",
                  "I-TARG:NEG", "B-TARG:NEUTRAL", "I-TARG:NEUTRAL"]
    else:
        labels = ["O", "B-TARG", "I-TARG"]

    label2idx = start_vocab(add_unk=False, add_padding=add_padding)
    index = len(label2idx)
    for label in labels:
        label2idx[label] = index
        index += 1

    return label2idx

def build_simple_rev_tag_vocab(tag_vocab):
    simple_rev_label_vocab = {}
    if len(tag_vocab) >= 7:
        for label, idx in tag_vocab.items():
            if label in ("O", PAD, UNK):
                simple_rev_label_vocab[idx] = ("O", "NEUTRAL")
            elif ":" in label:
                aspect, sent = label.split(":")
                if sent  == "POS":
                    simple_rev_label_vocab[idx] = (aspect, "POSITIVE")
                elif sent  == "NEG":
                    simple_rev_label_vocab[idx] = (aspect, "NEGATIVE")
                else:
                    simple_rev_label_vocab[idx] = (aspect, sent)
    return simple_rev_label_vocab

def build_joint_rev_tag_vocab(tag_vocab, class_vocab):
    joint_rev_label_vocab = {}
    for tag_label, tag_idx in tag_vocab.items():
        for class_label, class_idx in class_vocab.items():
            if tag_label in ("O", UNK, PAD):
                joint_rev_label_vocab[str((tag_idx, class_idx))] = "O"
            else:
                if class_label == "POSITIVE":
                    joint_rev_label_vocab[str((tag_idx, class_idx))] = tag_label+":POS"
                if class_label == "NEGATIVE":
                    joint_rev_label_vocab[str((tag_idx, class_idx))] = tag_label + ":NEG"
                else:
                    joint_rev_label_vocab[str((tag_idx, class_idx))] = tag_label + ":NEUTRAL"

    return joint_rev_label_vocab

def build_class_vocab(joint):
    if joint:
        class2idx = {"NEUTRAL": 0, "POS": 1, "NEG": 2}
    else:
        class2idx = {}
    return class2idx


# ----------------------------- PROCESS SENTENCES ---------------------------------------------------

def binarize_orientation(orientation):
    if orientation > 0:
        return "POS"
    elif orientation == 0:
        return "NEUTRAL"
    elif orientation < 0:
        return  "NEG"

def process_token(token):
    """Pre-processing for each token.
    As Liu et al. 2015 we lowercase all words
    and replace numbers with 'DIGIT'.
    Args:
        token (Token): Token
    Returns:
        str: processed token
    """
    ptoken = token.string.lower()
    if re.match("\\d+", ptoken):
        ptoken = "".join("DIGIT" for _ in ptoken)
    return ptoken


def sent_to_tokens(sentence, process_token_func, token2idx):
    sentence_seq = []
    for token in sentence.tokens:
        processed_token = process_token_func(token)
        sentence_seq.append(token2idx.get(processed_token, UNK_ID))
    return sentence_seq


def sent_to_iob_tags(sentence, sentiment=True, joint=False, strict=False):
    """Builds a list of aspect IOB tags based on the
    provided sentence.
    Args:
        sentence (Sentence): Sentence object
        joint: (bool)
        sentiment (bool): True if add sentiment (-1, 0 or 1) to the IOB label
    Returns:
        list: list of IOB tags
    """
    if not sentence.is_tokenized:
        raise Exception("Sentence not tokenized")

    if sentiment and joint:
        return Exception("Only choose sentiment or joint")

    tags = ["O"] * len(sentence)
    orientation = "NEUTRAL"

    if sentence.aspects:

        if (sentiment and strict) or joint \
        and len(set([binarize_orientation(a.orientation)
                     for a in sentence.aspects])) > 1:
            return ([], None)
        else:
            for aspect in sentence.aspects:
                orientation = binarize_orientation(int(aspect.orientation))
                for position in aspect.positions:
                    # only add aspects with known position
                    if position:
                        a_start, a_end = position
                        start = None
                        end = None
                        for token in sentence:
                            if token.start <= a_start < token.end:
                                start = token.index
                        for token in sentence:
                            if token.start < a_end <= token.end:
                                end = token.index
                        if None not in [start, end]:
                            if start == end:
                                tags[start] = "B-TARG"
                                if sentiment:
                                    tags[start] += ":" + orientation
                            elif end - start >= 1:
                                tags[start] = "B-TARG"
                                if sentiment:
                                    tags[start] += ":" + orientation
                                for i in range(start + 1, end + 1):
                                    tags[i] = "I-TARG"
                                    if sentiment:
                                        tags[i] += ":" + orientation
    if joint:
        result = (tags, orientation)
    else:
        result = (tags, None)

    return result

def sent_to_binary_features(sentence, funcs):
    """Generates binary one-hot features from a sentence.
    Applies each function in funcs to each token
    in the provided sentence.
    Args:
        sentence (Sentence):
            Sentence object
        funcs (list):
            list of functions to apply to
            each token in the sentence
    Returns:
        numpy.array : dim=(len(sentence), len(feat_funcs)
    """
    if not sentence.is_tokenized:
        raise Exception("Sentence not tokenized")
    matrix = np.zeros((len(sentence), len(funcs)))
    for i, token in enumerate(sentence):
        for j, func in enumerate(funcs):
            if func(token):
                matrix[i, j] = 1
    return matrix.tolist()

def sent_to_spectrogram(sentence, default_features=513):
    if sentence.captions:
        # time first
        spectrogram = sentence.spectrogram.T
    else:
        spectrogram = np.zeros((1, default_features))
    return spectrogram.tolist()


def build_sequences(sentences, token2idx, tag2idx, process_token_func,
                    class2idx=None, joint=False, sentiment=False, strict=False):
    sentence_id_seqs = []
    tag_id_seqs = []
    class_ids = []
    indices = []
    for i, sentence in enumerate(sentences):
        sentence_id_seq = sent_to_tokens(sentence, process_token_func, token2idx)
        tag_seq, class_ = sent_to_iob_tags(sentence, joint=joint, sentiment=sentiment, strict=strict)
        tag_id_seq = [tag2idx[tag] for tag in tag_seq]

        if not tag_seq:
            continue

        indices.append(i)
        sentence_id_seqs.append(sentence_id_seq)
        tag_id_seqs.append(tag_id_seq)
        if joint:
            class_ids.append(class2idx[class_])

    return sentence_id_seqs, tag_id_seqs, class_ids, indices

# ----------------------------- LIST SPLITTING -----------------------------------------------------------------

def split_list(examples, dev_ratio=0.9, test_ratio=None):
    """
    :param examples: list
    :param dev_ratio: 0.9
    :param test_ratio: to split train/test
    :param generate_test: (otherwise valid=test)
    :return:
    """
    if test_ratio:
        train, valid, test = [], [], []
        dev_size = int(len(examples)*test_ratio)
        train_size = int(dev_size * dev_ratio)
        for example in examples:
            r = random.random()
            if r <= test_ratio and len(train) + len(valid) < dev_size:
                rr = random.random()
                if rr < dev_ratio and len(train) < train_size:
                    train.append(example)
                else:
                    valid.append(example)
            else:
                test.append(example)
        return train, valid, test
    else:
        train, valid = [], []
        train_size = int(len(examples)*dev_ratio)
        for example in examples:
            r = random.random()
            if r <= dev_ratio and len(train) < train_size:
                train.append(example)
            else:
                valid.append(example)
    return train, valid

def kfolds_split_list(dataset, folds):
    """
    :param dataset:
    :param folds:
    :return: List of tuples.
    """
    result = []
    foldsize = 1.0 * len(dataset) / folds
    rest = 1.0 * len(dataset) % folds
    for f in range(1, folds + 1):
        start = (f - 1) * foldsize
        end = f * foldsize
        train_size = foldsize * (folds - 1)
        if f == folds:
            end += rest
        train, valid, test = [], [], []
        for i, example in enumerate(dataset):
            # test
            if start <= i < end:
                test.append(example)
            # development
            else:
                rr = random.random()
                if rr < 0.9 and len(train) < train_size:
                    train.append(example)
                else:
                    valid.append(example)
        result.append((train, valid, test))
    return result


def build_json_dataset(json_path, train_corpus, test_corpus=None, min_freq=1, feat_funcs=(), add_padding=True,
                       embeddings=None, test_ratio=0.8, folds=1, sentiment=False, joint=False, strict=False):
    """
    Generates JSON file/s for training a model on the provided corpus. If using folds,
    it generates a folder with the one JSON per fold.
    Args:
        json_path (str):
            Json path
        train_corpus (Corpus):
            Corpus object
        test_corpus (Corpus):
            Corpus object
        min_freq (int)
            Min frequency to add the token to the vocabulary.
        feat_funcs (list):
            list of functions to extract binary features from tokens, None if no
            binary features should be extracted (default=None)
        add_padding (bool)
            True to add the padding to the vocabulary.
        embeddings (Embeddings):
            Embeddings object
        test_ratio (float):
            Development/Test ratio when test set not given (default=0.8)
        folds (int)
            Use folds
        sentiment (bool):
            True if add sentiment (-1, 0 or 1) label to the aspect
        joint (bool)
            True if separate
    Returns:
        name (str) of the processed corpus if successful,
        or None if it fails
    """

    if not os.path.isdir(json_path):
        os.makedirs(json_path)

    if folds == 1:
        if test_corpus:
            assert train_corpus.pipeline == test_corpus.pipeline
            train, valid = split_list(train_corpus.sentences)
            partitions = [(train, valid, test_corpus.sentences)]
        else:
            partitions = [split_list(train_corpus.sentences, test_ratio=test_ratio)]
    else:
        partitions = kfolds_split_list(train_corpus.sentences, folds=folds)

    for f, partition in enumerate(partitions):

        if "SennaChunker" in train_corpus.pipeline:
            pipeline = "Senna"
        else:
            pipeline = "CoreNLP"

        jsondic = dict()
        train_sentences, valid_sentences, test_sentences = partition

        token2idx = build_token_vocab(train_sentences, process_token, min_freq=min_freq)
        tag2idx = build_tag_vocab(sentiment=sentiment)
        class2idx = build_class_vocab(joint)
        joint_rev_tag_vocab = build_joint_rev_tag_vocab(tag2idx, class2idx)
        simple_rev_tag_vocab = build_simple_rev_tag_vocab(tag2idx)

        train_x, train_y, train_z, train_indices = \
            build_sequences(train_sentences, token2idx, tag2idx, process_token,
                            class2idx=class2idx, joint=joint,
                            sentiment=sentiment, strict=strict)

        valid_x, valid_y, valid_z, valid_indices  = \
            build_sequences(valid_sentences, token2idx, tag2idx, process_token,
                            class2idx=class2idx, joint=joint,
                            sentiment=sentiment, strict=strict)

        test_x, test_y, test_z, test_indices = \
            build_sequences(test_sentences, token2idx, tag2idx, process_token,
                            class2idx=class2idx, joint=joint,
                            sentiment=sentiment, strict=strict)

        train_feat_x = [sent_to_binary_features(train_sentences[i], feat_funcs)
                        for i in train_indices]

        valid_feat_x = [sent_to_binary_features(valid_sentences[i], feat_funcs)
                        for i in valid_indices]

        test_feat_x = [sent_to_binary_features(test_sentences[i], feat_funcs)
                       for i in test_indices]


        if embeddings:
            vocab_size = len(token2idx)
            vector_size = embeddings.vector_size
            matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)

            # set the the unseen/unknown token embedding
            matrix[token2idx[UNK]] = embeddings.unseen()

            # set the the padding embedding
            if add_padding:
                matrix[token2idx[PAD]] = np.random.random((vector_size,))

            for token, idx in token2idx.items():
                if token in embeddings:
                    matrix[idx] = embeddings[token]
                else:
                    matrix[idx] = embeddings.unseen()

            jsondic["embeddings_matrix"] = matrix.tolist()
            embeddings_name = embeddings.name
        else:
            embeddings_name = "RandomEmbeddings"
            jsondic["embeddings_matrix"] = None

        jsondic["train_x"], jsondic["train_y"] = train_x, train_y
        jsondic["valid_x"], jsondic["valid_y"] = valid_x, valid_y
        jsondic["test_x"], jsondic["test_y"] = test_x, test_y
        jsondic["token2idx"] = token2idx
        jsondic["tag2idx"] = tag2idx
        jsondic["train_indices"] = train_indices
        jsondic["valid_indices"] = valid_indices
        jsondic["test_indices"] = test_indices
        jsondic["class2idx"] = class2idx
        jsondic["train_z"] = train_z
        jsondic["valid_z"] = valid_z
        jsondic["test_z"] = test_z
        jsondic["joint_rev_tag_vocab"] = joint_rev_tag_vocab
        jsondic["simple_rev_tag_vocab"] = simple_rev_tag_vocab
        jsondic["num_binary_features"] = len(feat_funcs)
        jsondic["train_feat_x"] = train_feat_x
        jsondic["valid_feat_x"] = valid_feat_x
        jsondic["test_feat_x"] = test_feat_x

        json_filename = pipeline + "." + train_corpus.name
        if test_corpus:
            json_filename += test_corpus.name
        json_filename += "." + embeddings_name

        json_filename += "." + str(min_freq)

        if sentiment:
            if strict:
                json_filename += ".strictsent"
            else:
                json_filename += ".sent"

        elif joint:
            json_filename += ".joint"
        else:
            json_filename += ".nosent"

        if folds > 1:
            jsondic["fold"] = f
            json_filename += "." + str(f)

        json_filename += ".json"

        with open(os.path.join(json_path, json_filename), "wb") as json_file:
            try:
                json.dump(jsondic, json_file)
            except Exception as e:
                warnings.warn(str(e))
            finally:
                print("Written " + json_filename)
