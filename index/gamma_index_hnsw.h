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
#include <vector>
#include <string>

#include "gamma_common_data.h"
#include "gamma_index_flat.h"
#include "log.h"
#include "field_range_index.h"
#include "bitmap.h"
#include "raw_vector.h"
#include "gamma_hnsw.h"

#include "faiss/impl/FaissException.h"
#include "faiss/IndexHNSW.h"

namespace tig_gamma {

struct GammaHNSWIndex : GammaFLATIndex, faiss::IndexHNSW {
	GammaHNSWIndex(faiss::Index *quantizer, size_t d,
                  DistanceMetricType metric_type,
                  int M, int efSearch, int efConstruction,
                  const char *docids_bitmap, RawVector<float> *raw_vec);

  ~GammaHNSWIndex();
  
  int Indexing() override;

  int AddRTVecsToIndex() override;

  bool Add(int n, const float *vec) override;

  int AddVertices(size_t n0, size_t n, const float *x,
                  bool verbose, bool preset_levels = false);

  int SearchHNSW(int n, const float *x, GammaSearchCondition *condition,
                 float *distances, idx_t *labels, int *total);

  int Search(const VectorQuery *query, 
            GammaSearchCondition *condition,
            VectorResult &result) override;

  long GetTotalMemBytes() override;

  DistanceComputer * GetDistanceComputer() const;
  
  int Update(int doc_id, const float *vec) override;

  int Delete(int doc_id);

  int Dump(const std::string &dir, int max_vid) override;

  int Load(const std::vector<std::string> &index_dirs) override;

  int indexed_vec_count_;
  GammaHNSW gamma_hnsw_;

  // each node have a lock for multi-thread add
  std::vector<omp_lock_t> locks_;

  // for search, every raw vector should be accessed
  float * raw_vec_head_;

  // for add and search
  pthread_rwlock_t mutex_;

#ifdef PERFORMANCE_TESTING
  int add_count_ = 0;
#endif
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
  struct GammaHNSWFlatIndex : GammaHNSWIndex {
    GammaHNSWFlatIndex(size_t d, DistanceMetricType metric_type,
                      int nlinks, int efSearch, int efConstruction,
                      const char *docids_bitmap, 
                      RawVector<float> *raw_vec);
};

}
#endif
