#!/bin/bash

ROOT=$(dirname "$PWD")

if [ ! -d "faiss" ]; then
  FAISS_HOME=$ROOT/third_party/faiss
  if [ $1 == "ON" ]; then
    pushd faiss-1.6.4
    CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake -B build -DFAISS_ENABLE_GPU=ON -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=${FAISS_HOME} .
    make -C build && make -C build install
    popd
  else
    tar -xzvf faiss-1.6.3.tar.gz
    mv faiss faiss-1.6.3
    pushd faiss-1.6.3
    ./configure --without-cuda --with-blas=/usr/lib64/libopenblas.so --prefix=${FAISS_HOME}
    make -j && make install
    popd
    \rm faiss/lib/libfaiss.so
    \rm -rf faiss-1.6.3
  fi
fi