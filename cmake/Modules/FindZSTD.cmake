# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# - Try to find Facebook zstd library
# This will define
# ZSTD_FOUND
# ZSTD_INCLUDE_DIR
# ZSTD_LIBRARY
#



SET(ZSTD_INCLUDE_SEARCH_PATHS
   $ENV{ZSTD_HOME}
   $ENV{ZSTD_HOME}/include
   $ENV{ZSTD_HOME}/lib
   /usr/include
   /usr/include/zfp
   /usr/local/include
   /usr/local/include/zfp   
)

SET(ZSTD_LIB_SEARCH_PATHS
    $ENV{ZSTD_HOME}
    $ENV{ZSTD_HOME}/lib
    $ENV{ZSTD_HOME}/lib64
    /lib/
    /lib64/
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64   
 )
 
FIND_PATH(ZSTD_INCLUDE_DIR NAMES zstd.h PATHS ${ZSTD_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(ZSTD_LIB NAMES zstd PATHS ${ZSTD_LIB_SEARCH_PATHS})

SET(ZSTD_FOUND ON)

# Check include files
IF(NOT ZSTD_INCLUDE_DIR)
  SET(ZSTD_FOUND OFF)
  MESSAGE(STATUS "Could not find ZSTD include. Turning ZSTD_FOUND off")
ENDIF()

# Check libraries
IF(NOT ZSTD_LIB)
  SET(ZSTD_FOUND OFF)
  MESSAGE(STATUS "Could not find ZSTD lib. Turning ZSTD_FOUND off")
ENDIF()


IF (ZSTD_FOUND)  
    MESSAGE(STATUS "Found ZSTD libraries: ${ZSTD_LIB}")
    MESSAGE(STATUS "Found ZSTD include: ${ZSTD_INCLUDE_DIR}")  
ELSE (ZSTD_FOUND)
  IF (ZSTD_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find zstd, please install ZSTD or set $ZSTD_HOME")
  ENDIF (ZSTD_FIND_REQUIRED)
ENDIF (ZSTD_FOUND)

MARK_AS_ADVANCED(
  ZSTD_INCLUDE_DIR
  ZSTD_LIB
  ZSTD
)
