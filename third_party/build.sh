#!/bin/bash

ROOT=$(dirname "$PWD")

if [ ! -d "faiss" ]; then
  FAISS_HOME=$ROOT/third_party/faiss
  pushd faiss-1.6.4
  cmake -B build -DFAISS_ENABLE_GPU=OFF -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=${FAISS_HOME} .
  make -C build && make -C build install
  popd
fi
