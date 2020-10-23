/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This faiss source code is licensed under the MIT license.
 * https://github.com/facebookresearch/faiss/blob/master/LICENSE
 *
 *
 * The works below are modified based on faiss:
 * 1. Replace the static batch indexing with real time indexing
 * 2. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2019 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef GAMMA_INDEX_HNSW_H_
#define GAMMA_INDEX_HNSW_H_

#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <mutex>

#include "bitmap.h"
#include "faiss/IndexHNSW.h"
#include "faiss/impl/FaissException.h"
#include "field_range_index.h"
#include "gamma_common_data.h"
#include "gamma_hnsw.h"
#include "gamma_index_flat.h"
#include "retrieval_model.h"
#include "log.h"
#include "raw_vector.h"
#include "gamma_index_io.h"

namespace tig_gamma {

class HnswRetrievalParameters : public RetrievalParameters {
 public:
  HnswRetrievalParameters() : RetrievalParameters() {
    efSearch_ = 64;
  }

  HnswRetrievalParameters(int efSearch,
                          enum DistanceComputeType type) {
    efSearch_ = efSearch;
    distance_compute_type_ = type;
  }

  HnswRetrievalParameters(enum DistanceComputeType type) {
    efSearch_ = 64;
    distance_compute_type_ = type;
  }

  ~HnswRetrievalParameters() {}

  int EfSearch() { return efSearch_; }
  void SetEfSearch(int efSearch) { 
      efSearch_ = efSearch; 
  }
 private:
  int efSearch_;
};

struct GammaHNSWIndex : public GammaFLATIndex, faiss::IndexHNSW {
  GammaHNSWIndex();

  GammaHNSWIndex(VectorReader *vec);

  ~GammaHNSWIndex();

  int Init(const std::string &model_parameters) override;

  RetrievalParameters *Parse(const std::string &parameters) override;

  int Indexing() override;

  bool Add(int n, const uint8_t *vec) override;

  int AddVertices(size_t n0, size_t n, const float *x, bool verbose,
                  bool preset_levels = false);

  int Update(const std::vector<int64_t> &ids,
             const std::vector<const uint8_t *> &vecs) override;

  int Delete(const std::vector<int64_t> &ids);

  int Search(RetrievalContext *retrieval_context, int n, const uint8_t *x,
             int k, float *distances, int64_t *labels);

  long GetTotalMemBytes() override;

  DistanceComputer *GetDistanceComputer(faiss::MetricType metric_type) const;

  int Dump(const std::string &dir) override;

  int Load(const std::string &index_dir) override;

  int indexed_vec_count_;
  GammaHNSW gamma_hnsw_;

  // each node have a lock for multi-thread add
  std::vector<omp_lock_t> locks_;

  // for add and search
  pthread_rwlock_t mutex_;
  
  // for dump
  std::mutex dump_mutex_;
  bool has_update_;

#ifdef PERFORMANCE_TESTING
  int add_count_ = 0;
#endif
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct GammaHNSWFlatIndex : GammaHNSWIndex {
	GammaHNSWFlatIndex();
};

}  // namespace tig_gamma
#endif
