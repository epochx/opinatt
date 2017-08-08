#!/usr/bin/python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from collections import OrderedDict
from os import path

from .utils import Corpus, CorpusError
from ..rep import Sentence, Document, Aspect
from ..settings import CORPORA_PATH

SEMEVAL_ABSA_2014_CORPORA_PATH = path.join(CORPORA_PATH, 'semeval-absa-2014')


class SemEvalABSA2014Corpus(Corpus):
    """
    Class to read SemEval ABSA corpus. Each sentence has an id, but there
    are no reviews. We assume it is only one review for all the sentences.
    """

    filepath = ""

    def __init__(self, filepath=None):
        if filepath:
            self.filepath=filepath
        # self.name = filepath.split('/')[-1].replace('.txt','')
        if self._check():
            self._read()
        else:
            raise CorpusError("Corpus was not properly built. Check for consistency")

    @property
    def aspects(self):
        return self._aspects.values()

    @property
    def sentences(self):
        return self._sentences.values()

    @property
    def reviews(self):
        return self._reviews.values()

    def __repr__(self):
        return "<SemEvalABSACorpus {0}>".format(self.name)

    def _check(self):
        return True

    def _read(self):
        self._counter = 1
        self._aspects = OrderedDict()
        self._reviews = OrderedDict()
        self._sentences = OrderedDict()

        # add the single fake review
        review = Document(id=1)
        self._reviews[1] = review

        tree = ET.parse(self.filepath)

        for xml_sentence in tree.getroot():
            sentence_id = xml_sentence.get("id")
            string = xml_sentence.find("text").text
            sentence = Sentence(string=string, id=sentence_id, document=review)
            sentence.aspects = []
            self._sentences[sentence_id] = sentence
            terms = xml_sentence.find("aspectTerms")
            if terms is not None:
                for term in terms:
                    term_string = term.get("term").strip()
                    orientation = term.get("polarity")
                    if orientation == 'positive':
                        orientation = 1
                    elif orientation == 'negative':
                        orientation = -1
                    else:
                        orientation = 0
                    position = (int(term.get("from")), int(term.get("to")))
                    ttype = "n"
                    aspect = self._aspects.get(term_string, None)
                    if aspect:
                        aspect.append(sentence, orientation,
                                      ttype, position=position)
                    else:
                        self._aspects[term_string] = Aspect(term_string,
                                                            sentence,
                                                            orientation,
                                                            ttype,
                                                            position=position)

LaptopsTrain = type("LaptopsTrain",
                    (SemEvalABSA2014Corpus,),
                    {'filepath': path.join(SEMEVAL_ABSA_2014_CORPORA_PATH,
                                                    'Laptop_Train_v2.xml')})

LaptopsTest = type("LaptopsTest",
                  (SemEvalABSA2014Corpus,),
                  {'filepath': path.join(SEMEVAL_ABSA_2014_CORPORA_PATH,
                                         'Laptops_Test_Data_phaseB.xml')})

RestaurantsTrain = type("RestaurantsTrain",
                       (SemEvalABSA2014Corpus,),
                       {'filepath': path.join(SEMEVAL_ABSA_2014_CORPORA_PATH,
                                              'Restaurants_Train_v2.xml')})

RestaurantsTest = type("RestaurantsTest",
                       (SemEvalABSA2014Corpus,),
                       {'filepath': path.join(SEMEVAL_ABSA_2014_CORPORA_PATH,
                                                      'Restaurants_Test_Data_phaseB.xml')})

__all__ = [LaptopsTrain,
           LaptopsTest,
           RestaurantsTrain,
           RestaurantsTest]
