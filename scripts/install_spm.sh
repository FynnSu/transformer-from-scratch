#!/bin/bash
# https://github.com/google/sentencepiece
# Download and install SPM

git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo update_dyld_shared_cache # For Mac users