#!/usr/bin/env bash

#!/bin/bash
DATAPATH=$1

mkdir -p "$DATAPATH"/data/{word_embeddings,pickle,json,results,corpus}
mkdir -p "$DATAPATH"/data/corpus/{semeval-absa-2014,youtube}

mv Laptop_Train_v2.xml "$DATAPATH"/data/corpus/semeval-absa-2014/
mv Restaurants_Train_v2.xml "$DATAPATH"/data/corpus/semeval-absa-2014/
mv Laptops_Test_Data_PhaseB.xml "$DATAPATH"/data/corpus/semeval-absa-2014/
mv Restaurants_Test_Data_PhaseB.xml "$DATAPATH"/data/corpus/semeval-absa-2014/
mv samsung_galaxy_s5.xml "$DATAPATH"/data/corpus/youtube/

mv ./deps.words.bz2 "$DATAPATH"/data/word_embeddings/deps.words.bz2
mv ./GoogleNews-vectors-negative300.bin.gz "$DATAPATH"/data/word_embeddings/GoogleNews-vectors-negative300.bin.gz
mv ~/senna/senna_embeddings.txt "$DATAPATH"/data/word_embeddings/senna_embeddings.txt