SET(FAISS_INCLUDE_SEARCH_PATHS
   /usr/include
   /usr/include/faiss
   /usr/local/include
   /usr/local/include/faiss
   $ENV{FAISS_HOME}
   $ENV{FAISS_HOME}/include
)

SET(FAISS_LIB_SEARCH_PATHS
    /lib/
    /lib64/
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{FAISS_HOME}
    $ENV{FAISS_HOME}/lib
 )

FIND_PATH(FAISS_INCLUDE_DIR NAMES faiss/Index.h PATHS ${FAISS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(FAISS_LIB NAMES faiss PATHS ${FAISS_LIB_SEARCH_PATHS})

SET(FAISS_FOUND ON)

# Check include files
IF(NOT FAISS_INCLUDE_DIR)
  SET(FAISS_FOUND OFF)
  MESSAGE(STATUS "Could not find Faiss include. Turning FAISS_FOUND off")
ENDIF()

# Check libraries
IF(NOT FAISS_LIB)
  SET(FAISS_FOUND OFF)
  MESSAGE(STATUS "Could not find Faiss lib. Turning FAISS_FOUND off")
ENDIF()

IF (FAISS_FOUND)
  IF (NOT FAISS_FIND_QUIETLY)
    MESSAGE(STATUS "Found Faiss libraries: ${FAISS_LIB}")
    MESSAGE(STATUS "Found Faiss include: ${FAISS_INCLUDE_DIR}")
  ENDIF (NOT FAISS_FIND_QUIETLY)
ELSE (FAISS_FOUND)
  IF (FAISS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find Faiss, please install faiss or set $FAISS_HOME")
  ENDIF (FAISS_FIND_REQUIRED)
ENDIF (FAISS_FOUND)

MARK_AS_ADVANCED(
  FAISS_INCLUDE_DIR
  FAISS_LIB
  FAISS
)

