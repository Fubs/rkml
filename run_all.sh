#!/bin/zsh
source ~/.zshrc

cd /home/max/Programming/phys/rkml/rk4
rm -rf data/*
make -j
./bin
cd ..
rm -rf dataprep/data
mkdir dataprep/data
rm -rf dataprep/csvfiles
mkdir dataprep/csvfiles
rm -rf dataprep/npzips
mkdir dataprep/npzips
rm -rf nnet/data/npzips
mkdir nnet/data/npzips
cp rk4/data/* dataprep/csvfiles
cd dataprep
expy makeGroupedNpzips.py
cd ..
cp dataprep/data/* nnet/data/npzips/
cd nnet
python run_net.py
