#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from enlp.pipe.chunkers import CoNLL2000Chunker, SennaChunker
from enlp.pipe.cparsers import CoreNLPConstParser
from enlp.corpus.semeval import LaptopsTrain, LaptopsTest, RestaurantsTrain, RestaurantsTest
from enlp.corpus.youtube import SamsungGalaxyS5

if __name__ == "__main__":

    Corpora = {"LaptopsTrain": LaptopsTrain,
               "LaptopsTest": LaptopsTest,
               "RestaurantsTrain": RestaurantsTrain,
               "RestaurantsTest": RestaurantsTest,
               "SamsungGalaxyS5": SamsungGalaxyS5}

    desc = "Help for process_datasets, a script that annotates a list of given corpora"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--corpus', "-c",
                        nargs='*',
                        choices=Corpora,
                        help="Names of corpus to use. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    annotators = ["Senna", "CoreNLP"]
    parser.add_argument('--annotators', "-a",
                        nargs='*',
                        choices=annotators,
                        help="Annotators to use to pre-process corpora. Allowed values are " + ', '.join(annotators),
                        metavar='')

    args = parser.parse_args()

    if args.corpus:
        corpus_names = args.corpus
    else:
        corpus_names = Corpora.keys()

    if args.annotators:
        annotator_names = args.annotators
    else:
        annotator_names = annotators

    for annotator_name in annotator_names:
        for corpus_name in corpus_names:
            Corpus = Corpora[corpus_name]
            corpus = Corpus()
            print "processing " + corpus.name
            if annotator_name == "CoreNLP":
                parser = CoreNLPConstParser()
                chunker = CoNLL2000Chunker()

                parser.batch_parse(corpus.sentences)
                chunker.batch_parse(corpus.sentences)
                corpus.freeze()

            if annotator_name == "Senna":
                chunker = SennaChunker()
                chunker.batch_parse(corpus.sentences)
                corpus.freeze()
