#!/bin/bash

ROOT=$(dirname "$PWD")

if [ ! -d "faiss" ]; then
  FAISS_HOME=$ROOT/third_party/faiss
  tar -xzvf faiss-1.6.3.tar.gz
  mv faiss faiss-1.6.3
  pushd faiss-1.6.3
  if [ -z $MKLROOT ]; then
    ./configure --without-cuda --prefix=${FAISS_HOME}
  else
    LDFLAGS=-L$MKLROOT/lib/intel64 ./configure --without-cuda --prefix=${FAISS_HOME}
  fi
  make -j4 && make install
  popd
  \rm faiss/lib/libfaiss.so
  \rm -rf faiss-1.6.3
fi
