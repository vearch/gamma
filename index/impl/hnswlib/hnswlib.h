/**
 * Copyright (c) Malkov, Yu A., and D. A. Yashunin.
 *
 * This hnswlib source code is licensed under the Apache-2.0 License.
 * https://github.com/nmslib/hnswlib/blob/master/LICENSE
 *
 *
 * The works below are modified based on hnswlib:
 * 1. Replace the static batch indexing with real time indexing
 * 2. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2020 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>

#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <string.h>

#include <iostream>
#include <queue>
#include <vector>

class RetrievalContext;

namespace hnswlib {
typedef size_t labeltype;

template <typename T>
class pairGreater {
 public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

template <typename MTYPE>
class SpaceInterface {
 public:
  // virtual void search(void *);
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE> get_dist_func() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

template <typename dist_t>
class AlgorithmInterface {
 public:
  virtual void addPoint(const void *datapoint, labeltype label) = 0;
  virtual std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
      const void *, size_t, DISTFUNC<dist_t>, size_t, int,
      const RetrievalContext *) = 0;
  template <typename Comp>
  std::vector<std::pair<dist_t, labeltype>> searchKnn(
      const void *, size_t, Comp, size_t, const RetrievalContext *) {}
  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

}  // namespace hnswlib

#include "hnswalg.h"
#include "space_ip.h"
#include "space_l2.h"
