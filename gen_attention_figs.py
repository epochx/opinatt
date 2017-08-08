#!/usr/bin/env python
# -*-coding: utf8 -*-

import json
import numpy as np
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="")

parser.add_argument("filepath",
                    help="Name of the attentions file")

parser.add_argument('save_path',
                    help="Path to save images")

args = parser.parse_args()

filename =os.path.basename(args.filepath)

alignments_dir = os.path.join(args.save_path, filename + ".alignments")

if not os.path.isdir(alignments_dir):
  os.makedirs(alignments_dir)

with open(args.filepath, "r") as f:
  data = json.load(f)

sequences = data["sequences"],
tagging_atts = data["tagging_atts"]
classification_atts = data["classification_atts"]
ref_tag_sequences = data["ref_tags"]
hyp_tags_sequences = data["hyp_tags"]
ref_classes = data["ref_classes"]
hyp_classes = data["hyp_classes"]

for i, sequence in enumerate(sequences[0]):

  try:
      tag_att = tagging_atts[i]
      ref_tags = ref_tag_sequences[i]
      hyp_tags = hyp_tags_sequences[i]
      tags = ["/".join((hyp, ref)) for hyp, ref in zip(hyp_tags, ref_tags)]
      name = os.path.join(alignments_dir, str(i) + ".tagging.png")
      fig = plt.figure(figsize=(5, 5))
      ax = fig.add_subplot(111)
      ax.matshow(tag_att, interpolation='nearest', aspect="auto", cmap=cm.gray)
      plt.xticks(np.arange(len(sequence)), sequence, rotation=45)
      plt.yticks(np.arange(len(sequence)), tags)
      plt.tight_layout()
      plt.savefig(name)
      plt.clf()
  except Exception as e:
      print e

  if classification_atts:
    try:
      class_att = classification_atts[i]
      fig = plt.figure(figsize=(5, 2))
      ax = fig.add_subplot(111)
      ax.matshow([class_att], interpolation='nearest', aspect="auto", cmap=cm.gray)
      plt.xticks(np.arange(len(sequence)), sequence, rotation=45)
      plt.yticks(np.arange(1), "")
      y_label = "/".join((hyp_classes[i], ref_classes[i]))
      plt.xlabel(y_label, labelpad=20)
      plt.tight_layout()
      plt.show()
      plt.savefig(name)
      plt.clf()
    except Exception as e:
      print e
