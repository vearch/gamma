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

#ifndef GAMMA_HNSW_H_
#define GAMMA_HNSW_H_

#include <pthread.h>

#include <queue>
#include <set>
#include <unordered_set>
#include <vector>

#include "bitmap.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/HNSW.h"
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"
#include "faiss/utils/hamming.h"
#include "field_range_index.h"
#include "log.h"
#include "utils.h"
#include "retrieval_model.h"

namespace tig_gamma {

using DistanceComputer = faiss::DistanceComputer;
using Node = faiss::HNSW::Node;

struct GammaHNSW : faiss::HNSW {
  GammaHNSW(int M);

  GammaHNSW();

  void AddLinksStartingFrom(DistanceComputer& ptdis, storage_idx_t pt_id,
                            storage_idx_t nearest, float d_nearest, int level,
                            omp_lock_t* locks);

  int AddWithLocks(DistanceComputer& ptdis, int pt_level, int pt_id,
                   std::vector<omp_lock_t>& locks);

  int SearchFromCandidates(DistanceComputer& qdis, int k, idx_t* I, float* D,
                           MinimaxHeap& candidates, int level, int efSearch,
                           const RetrievalContext *retrieval_context,
                           int nres_in = 0) const;

  std::priority_queue<Node> SearchFromCandidateUnbounded(
      const Node& node, DistanceComputer& qdis, size_t ef,
      const RetrievalContext *retrieval_context) const;

  void Search(DistanceComputer& qdis, int k, idx_t* I, float* D, int efSearch,
              const RetrievalContext *retrieval_context) const;

  int nlinks; 
  faiss::MetricType metric_type;
};
}  // namespace tig_gamma
#endif
