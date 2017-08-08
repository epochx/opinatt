#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from os import path
import xml.etree.ElementTree as ET

from ..rep import Sentence, Document, Aspect
from .semeval import SemEvalABSA2014Corpus
from ..settings import CORPORA_PATH


YOUTUBE_PATH = path.join(CORPORA_PATH, 'youtube')


class YoutubeCorpus(SemEvalABSA2014Corpus):

    def _read(self):
        self._counter = 1
        self._aspects = OrderedDict()
        self._reviews = OrderedDict()
        self._sentences = OrderedDict()

        tree = ET.parse(self.filepath)

        for xml_sentence in tree.getroot():
            sentence_id = xml_sentence.get("id")
            review_id = sentence_id.split(":")[0]
            review = self._reviews.get(review_id, None)
            if review is None:
                review = Document(id=review_id)
                self._reviews[review_id] = review
            string = xml_sentence.find("text").text
            sentence = Sentence(string=string, id=sentence_id, document=review)
            review.append(sentence)
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


SamsungGalaxyS5 = type("SamsungGalaxyS5",
                       (YoutubeCorpus,),
                       {"filepath": path.join(YOUTUBE_PATH,
                                              'samsung_galaxy_s5.xml')})
