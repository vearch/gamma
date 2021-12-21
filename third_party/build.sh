#!/bin/bash

ROOT=$(dirname "$PWD")

if [ ! -d "faiss" ]; then
  FAISS_HOME=$ROOT/third_party/faiss
  pushd faiss-1.6.4
  cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -D CMAKE_INSTALL_PREFIX=$FAISS_HOME -B build .
  make -C build && make -C build install
  popd
fi
