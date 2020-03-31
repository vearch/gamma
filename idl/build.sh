#!/bin/bash

path=`pwd`
FBS_GEN_PATH=$2

rm -rf $FBS_GEN_PATH
mkdir $FBS_GEN_PATH

THIRD_PARTY=$path/../third_party

FLATBUFFERS=$THIRD_PARTY/$1

$FLATBUFFERS/flatc -g -o $FBS_GEN_PATH/go $path/fbs/gamma_api.fbs
$FLATBUFFERS/flatc -c -o $FBS_GEN_PATH/c $path/fbs/gamma_api.fbs
