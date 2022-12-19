#!/bin/zsh
#source ~/.zshrc
#conda deactivate 1>/dev/null 2>&1

rm -rf dataprep/data
mkdir dataprep/data
rm -rf dataprep/csvfiles
mkdir dataprep/csvfiles
rm -rf dataprep/npzips
mkdir dataprep/npzips
rm -rf nnet/data/npzips
mkdir nnet/data/npzips
cp rk4/data/* dataprep/csvfiles
chdir dataprep
expy makeGroupedNpzips.py
chdir ..
cp dataprep/data/* nnet/data/npzips/

