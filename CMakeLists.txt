#--------------------------------------------
#    GAMMA ENGINE 
#--------------------------------------------
project(gamma_engine C CXX)
cmake_minimum_required(VERSION 3.0)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(FAISS REQUIRED)

option(BUILD_TEST "Build tests" off)
option(BUILD_WITH_GPU "Build gamma with gpu index support" off)
option(BUILD_TOOLS "Build tools" off)
option(SMALL_DOC_NUM_OPTIMIZATION "SMALL_DOC_NUM_OPTIMIZATION" off)

#ENV VARs
set(THIRDPARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(BTREE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/btree)
set(CJSON ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cjson)
set(EASYLOGGINGPP ${CMAKE_CURRENT_SOURCE_DIR}/third_party/easyloggingpp)
set(HASHTABLE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libcuckoo)
set(UTIL ${CMAKE_CURRENT_SOURCE_DIR}/util)
set(GAMMA_SEARCH ${CMAKE_CURRENT_SOURCE_DIR}/search)
set(GAMMA_INDEX ${CMAKE_CURRENT_SOURCE_DIR}/index)
set(GAMMA_VECTOR ${CMAKE_CURRENT_SOURCE_DIR}/vector)
set(GAMMA_REALTIME ${CMAKE_CURRENT_SOURCE_DIR}/realtime)
set(GAMMA_PROFILE ${CMAKE_CURRENT_SOURCE_DIR}/profile)
set(GAMMA_C_API ${CMAKE_CURRENT_SOURCE_DIR}/c_api)
set(FLATBUFFERS flatbuffers-1.11.0)
set(FLATBUFFERS_OUT ${CMAKE_CURRENT_SOURCE_DIR}/idl/fbs-gen)

if(APPLE)
    set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -mavx2 -msse4 -mpopcnt -fopenmp -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE -DOPEN_CORE -O0 -w -g3 -gdwarf-2")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -fPIC -m64 -Wall -O3 -mavx2 -msse4 -mpopcnt -fopenmp -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE -Werror=narrowing -Wno-deprecated")

if(DEFINED ENV{ROCKSDB_HOME})
    message(STATUS "RocksDB home is set=$ENV{ROCKSDB_HOME}")
    add_definitions(-DWITH_ROCKSDB)
    find_package(RocksDB REQUIRED)
else(DEFINED ENV{ROCKSDB_HOME})
    message(STATUS "RocksDB home isn't set, so store_type=RocksDB is not supported! ")
endif(DEFINED ENV{ROCKSDB_HOME})

#INCLUDE DIR
include_directories(
    ${THIRDPARTY}
    ${BTREE}
    ${CJSON}
    ${EASYLOGGINGPP}
    ${HASHTABLE}
    ${UTIL}
    ${GAMMA_PROFILE}
    ${GAMMA_SEARCH}
    ${GAMMA_INDEX}
    ${GAMMA_VECTOR}
    ${GAMMA_REALTIME}
    ${GAMMA_C_API}
    ${FAISS_INCLUDE_DIR}
    ${RocksDB_INCLUDE_DIR}
    ${FLATBUFFERS_OUT}/c
)

#LINK LIBRARY
set(LIBRARIES
    ${FAISS_LIB}
    ${RocksDB_LIB}
)

#ADDITIONAL SOURCE DIR
aux_source_directory(${THIRDPARTY} DIR_SRCS)
aux_source_directory(${BTREE} DIR_SRCS)
aux_source_directory(${CJSON} DIR_SRCS)
aux_source_directory(${EASYLOGGINGPP} DIR_SRCS)
aux_source_directory(${UTIL} DIR_SRCS)
aux_source_directory(${GAMMA_SEARCH} DIR_SRCS)
aux_source_directory(${GAMMA_INDEX} DIR_SRCS)
aux_source_directory(${GAMMA_VECTOR} DIR_SRCS)
aux_source_directory(${GAMMA_REALTIME} DIR_SRCS)
aux_source_directory(${GAMMA_PROFILE} DIR_SRCS)
aux_source_directory(${GAMMA_C_API} DIR_SRCS)

set(LIB_SRCS ${DIR_SRCS})

add_definitions(-DHAVE_NETINET_IN_H)
add_definitions(-DFINTEGER=int)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug Mode")
    message(STATUS "Flags:" ${CMAKE_CXX_FLAGS_DEBUG})
    add_definitions(-DDEBUG_) 
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
    message(STATUS "Release Mode")
    message(STATUS "Flags:" ${CMAKE_CXX_FLAGS_RELEASE})
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "log --format='[sha1]:%h [author]:%cn [time]:%ci [branch]:%d' -1"
    OUTPUT_VARIABLE VERSION_SHA1)

add_definitions(-DGIT_SHA1="${VERSION_SHA1}")

if(NOT EXISTS "${THIRDPARTY}/${FLATBUFFERS}")
    execute_process(COMMAND wget https://github.com/google/flatbuffers/archive/v1.11.0.tar.gz WORKING_DIRECTORY ${THIRDPARTY})
    execute_process(COMMAND tar xf v1.11.0.tar.gz WORKING_DIRECTORY ${THIRDPARTY})
    execute_process(COMMAND cmake . WORKING_DIRECTORY ${THIRDPARTY}/${FLATBUFFERS})
    execute_process(COMMAND make WORKING_DIRECTORY ${THIRDPARTY}/${FLATBUFFERS})
    execute_process(COMMAND rm -rf v1.11.0.tar.gz WORKING_DIRECTORY ${THIRDPARTY})
    execute_process(COMMAND rm -rf flatbuffers WORKING_DIRECTORY ${THIRDPARTY})
    exec_program(
        "cp"
        ${THIRDPARTY}
        ARGS "-r -p ${FLATBUFFERS}/include/flatbuffers .")
    exec_program(
        "sh"
        ${CMAKE_CURRENT_SOURCE_DIR}/idl
        ARGS "build.sh ${FLATBUFFERS} ${FLATBUFFERS_OUT}")
endif()

if (PERFORMANCE_TESTING STREQUAL "ON")
    message(STATUS "With preformance testing")
    add_definitions(-DPERFORMANCE_TESTING) 
endif(PERFORMANCE_TESTING STREQUAL "ON")

if (SMALL_DOC_NUM_OPTIMIZATION)
    message(STATUS "with small doc num optimization")
    add_definitions(-DSMALL_DOC_NUM_OPTIMIZATION)
endif(SMALL_DOC_NUM_OPTIMIZATION)

if(BUILD_WITH_GPU)
    message(STATUS "With GPU")
    add_definitions(-DBUILD_GPU) 
    
    set(GAMMA_GPU ${CMAKE_CURRENT_SOURCE_DIR}/index/gpu)

    aux_source_directory(${GAMMA_GPU} DIR_SRCS)

    include(cmake/Modules/gpu/CUDA.cmake)
    
    file(GLOB_RECURSE FAISS_GPU_HEADERS ${GAMMA_GPU}/*.h)
    file(GLOB_RECURSE FAISS_GPU_CPP ${GAMMA_GPU}/*.cpp)
    file(GLOB_RECURSE FAISS_GPU_CC ${GAMMA_GPU}/*.cc)
    file(GLOB_RECURSE FAISS_GPU_CUH ${GAMMA_GPU}/*.cuh)
    file(GLOB_RECURSE FAISS_GPU_CU ${GAMMA_GPU}/*.cu)

    include_directories(
        ${GAMMA_GPU}
        ${FAISS_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
    )

    CUDA_ADD_LIBRARY(gamma SHARED 
    ${FAISS_GPU_HEADERS}
    ${FAISS_GPU_CPP}
    ${FAISS_GPU_CC}
    ${FAISS_GPU_CUH}
    ${FAISS_GPU_CU}
    ${LIB_SRCS})
else (BUILD_WITH_GPU)
    add_library(gamma SHARED ${LIB_SRCS})
endif(BUILD_WITH_GPU)

set(LIB_SRCS ${DIR_SRCS})

target_link_libraries(gamma ${LIBRARIES})

set_target_properties(gamma PROPERTIES VERSION 0.1 SOVERSION 0.1)

install(TARGETS gamma
	LIBRARY	DESTINATION	lib 
	ARCHIVE	DESTINATION	lib)

install(FILES ${GAMMA_C_API}/gamma_api.h DESTINATION include)

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif(BUILD_TEST)

if(BUILD_TOOLS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
endif(BUILD_TOOLS)

