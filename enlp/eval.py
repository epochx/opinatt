#!/usr/bin/python
# -*- coding: utf-8 -*-

from .settings import CONLLEVAL_PATH
import re
import subprocess


def classes2class(classes):
  if "POSITIVE" in classes:
    return "POSITIVE"
  elif "NEGATIVE" in classes:
    return "NEGATIVE"
  else:
    return "NEUTRAL"


def tagclasses2classes(sentence_classes, sentence_tags):
  # we only rescue the first letter of each tag (I/O/B)
  classes = []
  regex = "(BI*)"
  sentence_tags_str = "".join([tag[0] for tag in sentence_tags])
  for match in re.finditer(regex, sentence_tags_str):
    aspect_classes = sentence_classes[match.start(): match.end()]
    aspect_class = classes2class(aspect_classes)
    classes.append(aspect_class)
  return classes

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  if v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def conlleval(p, g, w, filename):
    """
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)


def get_perf(filename):
    """
    run conlleval.pl perl script to obtain
    precision/recall and F1 score
    """

    proc = subprocess.Popen(["perl", CONLLEVAL_PATH],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


class ConfusionMatrix(object):

    def __init__(self, label, true_positives=None, false_positives=None, true_negatives=None,
                 false_negatives=None):
        self.label = label
        self.tp = self.true_positives = true_positives if true_positives else []
        self.fp = self.false_positives = false_positives if false_positives else []
        self.tn = self.true_negatives = true_negatives if true_negatives else []
        self.fn = self.false_negatives = false_negatives if false_negatives else[]

    def __repr__(self, *args, **kwargs):
        string = ''
        string += "True Positives: " + str(len(self.tp)) + '\n'
        string += "False Positives: " + str(len(self.fp)) + '\n'
        string += "True Negatives: " + str(len(self.tn)) + '\n'
        string += "False negatives: " + str(len(self.fn)) + '\n'
        return string

    @property
    def precision(self):
        try:
            return 1.0 * len(self.tp) / (len(self.tp) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def recall(self):
        try:
            return 1.0 * len(self.tp) / (len(self.tp) + len(self.fn))
        except ZeroDivisionError:
            return 0

    @property
    def fmeasure(self):
        try:
            precision = self.precision
            recall = self.recall
            return 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            return 0

    @property
    def accuracy(self):
        try:
            return 1.0 * (len(self.tp) + len(self.tn)) / (len(self.tp) + len(self.tn) + len(self.fn) + len(self.fp))
        except ZeroDivisionError:
            return 0

    @property
    def measures(self):
        return {"p": self.precision,
                "r": self.recall,
                "f1": self.fmeasure,
                "a": self.accuracy}

    p = precision
    r = recall
    a = accuracy
    f = fmeasure


def classeval(predicted_labels, reference_labels, labels, classification_out_file=None):
    assert len(predicted_labels) == len(reference_labels)

    cm = {}
    for label in labels:
        cm[label] = ConfusionMatrix(label)

    cm["all"] = ConfusionMatrix("all")

    for i, (pred_label, ref_label) in enumerate(zip(predicted_labels, reference_labels)):
        if pred_label == ref_label:
            cm[ref_label].tp.append(i)
            cm["all"].tp.append(i)
            for label in labels:
                if label != ref_label:
                    cm[label].tn.append(i)
        else:
            cm[ref_label].fn.append(i)
            cm[pred_label].fp.append(i)
            cm["all"].fp.append(i)

    if classification_out_file:
        with open(classification_out_file, "w") as f:
            for pred_label, ref_label in zip(predicted_labels, reference_labels):
                f.write("{0} {1}\n".format(pred_label, ref_label))

    return {key:value.measures for key, value in cm.items()}