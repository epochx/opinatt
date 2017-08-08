
#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import os
from platform import architecture, system


class DepParser(object):

    def __init__(self, ssc):
        raise NotImplemented

    def parse(self):
        raise NotImplemented

    def batch_parse(self):
        raise NotImplemented

class CoreNLPDepParser(Parser):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs["ssplit_isOneSentence"] = "True"
        """Wrapper for the StanfordCoreNLP Parser"""
        self.parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", 
                                          "lemma", "parse", "depparse"],
                              **self.kwargs)

    def parse(self, sentence, build=False):

        if isinstance(sentence, basestring):
            # get only first document
            parsed_sentence = self.parser.parse(sentence)[0]
            if build:
                return self._process_sentence(sentence, parsed_sentence)
            else:
                return parsed_sentence

        elif isinstance(sentence, Sentence):
            if sentence.is_tokenized:
                if "tokenize_whitespace" not in self.kwargs:
                    raise Exception("Add the tokenize_whitespace kwarg please")
            # get only first document
            parsed_sentence = self.parser.parse(sentence.string)[0]
            self._process_sentence(sentence, parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def batch_parse(self, sentences, build=False):
        if isinstance(sentences[0], basestring):
            parsed_sentences = [result[0] for result in
                                self.parser.batch_parse(sentences)]

            if build:
                return [self._process_sentence(sentences[i], parsed_sentence)
                        for i, parsed_sentence in enumerate(parsed_sentences)]
            else:
                return parsed_sentences

        elif isinstance(sentences[0], Sentence):
            if any([s.is_tokenized for s in sentences]):
                if "tokenize_whitespace" not in self.kwargs:
                    raise Exception("Add the tokenize_whitespace kwarg please")
                else:
                    strings = [" ".join([t.string for t in sentence])
                               for sentence in sentences]
            else:
                strings = [sentence.string for sentence in sentences]
            # get only first document for each sentence
            parsed_sentences = [result[0] for result in
                                self.parser.batch_parse(strings)]
            for i, parsed_sentence in enumerate(parsed_sentences):
                self._process_sentence(sentences[i], parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def _process_sentence(self, sentence, parsed_sentence):
        """
        Process a raw Stanford parsed sentence (parsed_sentence)
        and add tokens to sentence.
        """
        should_return = False
        if isinstance(sentence, basestring):
            sentence = Sentence(string=sentence)
            should_return = True

        if not sentence.is_tokenized:
            for token in parsed_sentence.tokens:
                sentence.append(string=token.word,
                                start=token.start,
                                end=token.end,
                                lemma=token.lemma,
                                pos_tag=token.POS)
        else:
            if not sentence.is_tagged:
                tags = [t.POS for t in parsed_sentence.tokens]
                sentence.append_tags(pos_tags=tags)

            if not sentence.is_lemmatized:
                lemmas = [t.lemma for t in parsed_sentence.tokens]
                sentence.append_tags(lemmas=lemmas)

        deps = []

        for dep in parsed_sentence.dependencies:
            head_index = int(dep.head.index)-1
            dep_index = int(dep.dependent.index)-1
            deps.append((head_index, dep.label, dep_index))

        if deps:
            sentence.append_tags(rels=deps)

        # eliminate the ROOT node
        syntax_tree = parsed_sentence.syntax_tree

        if "(ROOT " in syntax_tree:
            syntax_tree = syntax_tree.strip().replace("(ROOT ", "")[:-1]

        sentence.tree = syntax_tree
        sentence.pipeline.append(str(self))

        if should_return:
            return sentence




