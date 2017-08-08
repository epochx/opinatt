#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path

CODE_ROOT = path.dirname(path.realpath(__file__))

# Environment PATH settings
PAD_ID = 0
UNK_ID = 1

PAD = "PAD"
UNK = "UNK"

DATA_PATH = "/home/ubuntu/data"
CORENLP_PATH = "/home/ubuntu/stanford-corenlp-full-2015-12-09"
JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64"
SENNA_PATH = "/home/ubuntu/senna"

CORPORA_PATH = path.join(DATA_PATH, "corpus")
CHUNKLINK_PATH = path.join(CODE_ROOT, "script/mod_chunklink_2-2-2000_for_conll.pl")
CONLLEVAL_PATH = path.join(CODE_ROOT, "script/conlleval.pl")
PICKLE_PATH = path.join(DATA_PATH, "pickle")
