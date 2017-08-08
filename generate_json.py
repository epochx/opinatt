import os
import argparse
from enlp.corpus.youtube import SamsungGalaxyS5
from enlp.corpus.semeval import LaptopsTest, LaptopsTrain, RestaurantsTest, RestaurantsTrain
from enlp.embeddings import GoogleNews, SennaEmbeddings, WikiDeps
from enlp.data import build_json_dataset

if __name__ == "__main__":

    Corpora = {"LaptopsTrain": LaptopsTrain,
               "LaptopsTest": LaptopsTest,
               "RestaurantsTrain": RestaurantsTrain,
               "RestaurantsTest": RestaurantsTest,
               "SamsungGalaxyS5": SamsungGalaxyS5}

    desc = "Help for build_json_fold_datasets, a script that takes processed corpora and " \
           "embeddings and generates JSON files for training attenttion-RNNs for aspect-based " \
           "opinion mining using k-fold cross validation"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--json_path",
                        required=True,
                        help="Absolute path to store JSON files" )

    parser.add_argument("--sentiment", "-s",
                        action='store_true',
                        help="To generate IOB tags that include sentiment (TARG+/TARG-/TARG0)")

    parser.add_argument("--joint", "-j",
                        action='store_true',
                        help="To generate IOB tags and classes")

    parser.add_argument("--strict", "-st",
                        action='store_true',
                        help="When using joint, use only single aspect or single sentiment sentences")

    parser.add_argument("--padding", "-p",
                        action='store_true',
                        default=True,
                        help="Add padding. Default=True")

    parser.add_argument("--minfreq", "-mf",
                        type=int,
                        default=1,
                        help="Minimum frequency (default=1)")

    parser.add_argument('--train', "-tr",
                        nargs='?',
                        choices=Corpora,
                        help="Corpus to use as training. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    parser.add_argument('--test', "-ts",
                        nargs='?',
                        choices=Corpora,
                        help="Corpus to use as test. Allowed values are " + ', '.join(Corpora),
                        metavar='')

    embs = ["GoogleNews", "SennaEmbeddings", "WikiDeps"]

    parser.add_argument('--embeddings', "-e",
                        nargs='*',
                        choices=embs,
                        help="Names of embeddings to use. Allowed values are " + ', '.join(embs),
                        metavar='')

    parser.add_argument('--folds', "-f",
                        type=int,
                        default=1,
                        help="Number of folds to use. Default, no folds (1)")

    parser.add_argument('--ratio', "-r",
                        type=float,
                        default=0.8,
                        help="Train/Test ratio when test set not given. Default=0.8")

    feat_funcs = [lambda t: "JJ" in t.pos,
                  lambda t: "NN" in t.pos,
                  lambda t: "RB" in t.pos,
                  lambda t: "VB" in t.pos,
                  lambda t: t.iob == "B-NP",
                  lambda t: t.iob == "B-PP",
                  lambda t: t.iob == "B-VP",
                  lambda t: t.iob == "B-ADJP",
                  lambda t: t.iob == "B-ADVP",
                  lambda t: t.iob == "I-NP",
                  lambda t: t.iob == "I-PP",
                  lambda t: t.iob == "I-VP",
                  lambda t: t.iob == "I-ADJP",
                  lambda t: t.iob == "I-ADVP"]

    args = parser.parse_args()

    if args.joint and args.sentiment:
        raise KeyError("Please choose --sentiment or --joint")

    if not os.path.exists(args.json_path):
        print("Creating " + args.json_path)
        os.makedirs(args.json_path)

    if args.embeddings:
        embeddings_list = []
        if "GoogleNews" in args.embeddings:
            embeddings_list.append(GoogleNews)
        if "SennaEmbeddings" in args.embeddings:
            embeddings_list.append(SennaEmbeddings)
        if "WikiDeps" in args.embeddings:
            embeddings_list.append(WikiDeps)
    else:
        embeddings_list = [None]

    for Embeddings in embeddings_list:
        if Embeddings:
            print("loading " + str(Embeddings.__name__))
            embeddings = Embeddings()
        else:
            embeddings = Embeddings

        TrainCorpus = Corpora[args.train]
        TestCorpus = Corpora.get(args.test, None)
        for pipeline in TrainCorpus.list_frozen():
            if any(["Chunker" in item for item in pipeline]):
                train_corpus = TrainCorpus.unfreeze(pipeline)
                print("Using " + train_corpus.name + " " + str(train_corpus.pipeline))
                if TestCorpus:
                    for pipeline in TestCorpus.list_frozen():
                        if any(["Chunker" in item for item in pipeline]) \
                        and pipeline == train_corpus.pipeline:
                            test_corpus = TestCorpus.unfreeze(pipeline)
                            print("Using " + test_corpus.name + " " + str(test_corpus.pipeline))
                            build_json_dataset(args.json_path,
                                               train_corpus, test_corpus=test_corpus, min_freq=args.minfreq,
                                               add_padding=args.padding, feat_funcs=feat_funcs,
                                               embeddings=embeddings, sentiment=args.sentiment,
                                               joint=args.joint, strict=args.strict)
                else:
                    build_json_dataset(args.json_path, train_corpus, min_freq=args.minfreq,
                                       add_padding=args.padding, feat_funcs=feat_funcs, embeddings=embeddings,
                                       sentiment=args.sentiment, test_ratio=args.ratio, folds=args.folds,
                                       joint=args.joint, strict=args.strict)