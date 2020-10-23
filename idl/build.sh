#!/bin/bash

BASE_PATH=`pwd`
FBS_GEN_PATH=fbs-gen

THIRD_PARTY=$BASE_PATH/../third_party

if [ ! -d "$THIRD_PARTY/flatbuffers-1.11.0" ]
then
  cd $THIRD_PARTY
  wget https://github.com/google/flatbuffers/archive/v1.11.0.tar.gz
  tar xf v1.11.0.tar.gz
  rm -rf v1.11.0.tar.gz
  cd flatbuffers-1.11.0
  cmake . && make -j
  cd ..
  rm -rf flatbuffers
  cp -r -p flatbuffers-1.11.0/include/flatbuffers .
fi

cd $BASE_PATH

rm -rf $FBS_GEN_PATH
mkdir $FBS_GEN_PATH

FLATBUFFERS=$THIRD_PARTY/flatbuffers-1.11.0

$FLATBUFFERS/flatc -g -o $FBS_GEN_PATH/go $BASE_PATH/fbs/*.fbs --go-namespace gamma_api
$FLATBUFFERS/flatc -c --no-prefix -o $FBS_GEN_PATH/c $BASE_PATH/fbs/*.fbs
$FLATBUFFERS/flatc -p -o $FBS_GEN_PATH/python $BASE_PATH/fbs/*.fbs
