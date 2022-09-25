#!/bin/bash
mkdir data
cd data

wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
wget 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'
for YEAR in 2012 2013 2014; do
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.en"
    wget "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest${YEAR}.de"
done
cat newstest{2012,2013}.en >dev.en
cat newstest{2012,2013}.de >dev.de
cp newstest2014.en test.en
cp newstest2014.de test.de

cat train.en train.de > train.en-de
spm_train --input=train.en-de --model_prefix=bpe --vocab_size=37000 --character_coverage=1.0 --model_type=bpe

for SET in train dev test; do
    python ../src/preprocess.py --src_path $SET.en --tgt_path $SET.de --model_path bpe.model
done