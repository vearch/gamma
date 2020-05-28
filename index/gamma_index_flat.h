/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This faiss source code is licensed under the MIT license.
 * https://github.com/facebookresearch/faiss/blob/master/LICENSE
 *
 *
 * The works below are modified based on faiss:
 * 1. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2019 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#ifndef GAMMA_INDEX_FLAT_H_
#define GAMMA_INDEX_FLAT_H_

#include <vector>
#include <string>

#include "bitmap.h"
#include "utils.h"
#include "field_range_index.h"
#include "gamma_common_data.h"
#include "gamma_index.h"
#include "log.h"
#include "raw_vector.h"

#include "omp.h"

#include "faiss/impl/FaissAssert.h"
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"
#include "faiss/utils/hamming.h"
#include "faiss/utils/utils.h"
#include "faiss/Index.h"

namespace tig_gamma {
using idx_t = faiss::Index::idx_t;

struct GammaFLATIndex : GammaIndex {
	GammaFLATIndex(size_t d, const char *docids_bitmap, RawVector<float> *raw_vec);

  ~GammaFLATIndex();
  
  int Indexing() override;

  int AddRTVecsToIndex() override;

  bool Add(int n, const float *vec) override;

  int Search(const VectorQuery *query, 
            GammaSearchCondition *condition,
            VectorResult &result) override;

  void SearchDirectly(int n, const float *x,
                     GammaSearchCondition *condition,
                     float *distances, idx_t *labels,
                     int *total);

  int Delete(int doc_id) { return 0; }

  long GetTotalMemBytes() override;

  int Update(int doc_id, const float *vec) override;

  int Dump(const std::string &dir, int max_vid) override;

  int Load(const std::vector<std::string> &index_dirs) override;

  faiss::MetricType metric_type_;
};

}
#endif
