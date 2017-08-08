#!/usr/bin/python
# -*- coding: utf-8 -*-


from ..rep import Sentence
from ..corenlp import CoreNLP
from ..senna import Senna


class Parser():

    def __str__(self):
        return self.__class__.__name__

    def parse(self):
        raise NotImplementedError

    def batch_parse(self):
        raise NotImplementedError


class CoreNLPConstParser(Parser):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs["ssplit_isOneSentence"] = "True"
        """Wrapper for the StanfordCoreNLP Parser"""
        self.parser = CoreNLP(annotators=["tokenize", "ssplit", "pos", "lemma", "parse"], **self.kwargs)

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


class SennaParser(Parser):

    def __init__(self, args=[]):
        self.parser = Senna()
        self.args = ["pos", "chk", "psg", "iobtags"] + args

    def parse(self, sentence, build=False):
        if isinstance(sentence, basestring):
            # get only first document
            parsed_sentence = self.parser.parse(sentence,
                                                args=self.args)
            if build:
                return self._process_sentence(sentence, parsed_sentence)
            else:
                return parsed_sentence

        elif isinstance(sentence, Sentence):
            if sentence.is_tokenized:
                if "usrtokens" not in self.args:
                    raise Exception("Add usrtokens to args please")
                else:
                    string = " ".join(token.string for token in sentence)
                # get only first document
            else:
                string = sentence.string
            parsed_sentence = self.parser.parse(string, args=self.args)
            self._process_sentence(sentence, parsed_sentence)
        else:
            raise TypeError("Sentence type nor supported")

    def batch_parse(self, sentences, build=False):
        if isinstance(sentences[0], basestring):
            parsed_sentences = self.parser.batch_parse(sentences)
            if build:
                return [self._process_sentence(sentences[i], parsed_sentence)
                        for i, parsed_sentence in enumerate(parsed_sentences)]
            else:
                return parsed_sentences

        elif isinstance(sentences[0], Sentence):
            if any([s.is_tokenized for s in sentences]):
                if "usrtokens" not in self.args:
                    raise Exception("Add usrtokens to args please")
                else:
                    strings = [" ".join([t.string for t in sentence])
                               for sentence in sentences]
            else:
                strings = [sentence.string for sentence in sentences]
            # get only first document for each sentence
            parsed_sentences = self.parser.batch_parse(strings)
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
                                pos_tag=token.pos,
                                iob_tag=token.chunk)
        else:
            if not sentence.is_tagged:
                tags = [t.pos for t in parsed_sentence.tokens]
                sentence.append_tags(pos_tags=tags)

            if not sentence.is_chunked:
                iob_tags = [t.chunk for t in parsed_sentence.tokens]
                sentence.append_tags(iob_tags=iob_tags)

        syntax_tree = parsed_sentence.syntax_tree
        if "(S1" in syntax_tree:
            syntax_tree = syntax_tree.replace("(S1", "")[:-1]

        sentence.tree = syntax_tree
        sentence.pipeline.append(str(self))

        if should_return:
            return sentence


# class NLTKMaltParser():
#
#     os.environ["MALT_PARSER"] = "/home/edison/nltk_data/maltparser" # Name has to be: malt.jar"
#
#     def __init__(self):
#         raise NotImplementedError("Not functional yet.")
#         self.parser = MaltParser(working_dir = "/home/edison/nltk_data/maltparser", mco = "engmalt.linear-1.7" )
#         print(self.parser.working_dir)
#         #additional_java_args=['-Xmx512m']
#
#     def parse(self, sentence):
#         """ Use MaltParser to parse a sentence. Takes a sentence as a list of (word, tag) tuples; the sentence must have already been tokenized and tagged."""
#         return self.parser.tagged_parse(sentence)
#
#     def parse_sentences(self, sentences):
#         return self.parser.tagged_parse_sents(sentences)
#
#     def train_from_file(self, conll_file):
#         self.parser.train_from_file(conll_file)
