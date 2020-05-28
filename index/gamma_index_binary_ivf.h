/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexBinaryIVF.h>
#include <faiss/utils/utils.h>

#include <atomic>

#include "gamma_index.h"
#include "raw_vector.h"
#include "realtime_invert_index.h"

namespace tig_gamma {

using idx_t = faiss::Index::idx_t;

struct GammaBinaryInvertedListScanner {
  GammaBinaryInvertedListScanner() {
    docids_bitmap_ = nullptr;
    raw_vec_ = nullptr;
    range_index_ptr_ = nullptr;
  }

  /// from now on we handle this query.
  virtual void set_query(const uint8_t *query_vector) = 0;

  /// following codes come from this inverted list
  virtual void set_list(idx_t list_no, uint8_t coarse_dis) = 0;

  /// compute a single query-to-code distance
  // virtual uint32_t distance_to_code(const uint8_t *code) const = 0;

  /** compute the distances to codes. (distances, labels) should be
   * organized as a min- or max-heap
   *
   * @param n      number of codes to scan
   * @param codes  codes to scan (n * code_size)
   * @param ids        corresponding ids (ignored if store_pairs)
   * @param distances  heap distances (size k)
   * @param labels     heap labels (size k)
   * @param k          heap size
   */
  virtual size_t scan_codes(size_t n, const uint8_t *codes, const idx_t *ids,
                            int32_t *distances, idx_t *labels,
                            size_t k) const = 0;

  virtual ~GammaBinaryInvertedListScanner() {}

  inline void SetVecFilter(const char *docids_bitmap,
                           const RawVector<uint8_t> *raw_vec) {
    if (docids_bitmap == nullptr) {
      LOG(ERROR) << "docids_bitmap is NULL!";
      return;
    }

    if (!docids_bitmap_) {
      docids_bitmap_ = docids_bitmap;
    }

    if (!raw_vec_) {
      raw_vec_ = raw_vec;
    }

    return;
  }

  void set_search_condition(const GammaSearchCondition *condition) {
    range_index_ptr_ = condition->range_query_result;
  }

  const char *docids_bitmap_;
  const RawVector<uint8_t> *raw_vec_;
  MultiRangeQueryResults *range_index_ptr_;
};

class GammaIndexBinaryIVF : GammaIndex, faiss::IndexBinaryIVF {
 public:
  GammaIndexBinaryIVF(faiss::IndexBinary *quantizer, size_t d, size_t nlist,
                      size_t nprobe, const char *docids_bitmap,
                      RawVector<uint8_t> *raw_vec);

  GammaIndexBinaryIVF() = default;

  virtual ~GammaIndexBinaryIVF();

  int Indexing() override;

  int AddRTVecsToIndex() override;

  bool Add(int n, const uint8_t *vec);

  int Search(const VectorQuery *query, GammaSearchCondition *condition,
             VectorResult &result) override;

  long GetTotalMemBytes();

  int Dump(const std::string &dir, int max_vid) { return 0; }
  int Load(const std::vector<std::string> &index_dirs) { return 0; }

  int Delete(int doc_id);

 private:
  // assign the vectors, then call search_preassign
  void SearchHamming(int n, const uint8_t *x, GammaSearchCondition *condition,
                     int32_t *distances, idx_t *labels, int *total);

  void search_knn_hamming_heap(
      size_t n, const uint8_t *x, GammaSearchCondition *condition,
      const idx_t *keys, const int32_t *coarse_dis, int32_t *distances,
      idx_t *labels, bool store_pairs,
      const faiss::IVFSearchParameters *params = nullptr);

  void search_preassigned(int n, const uint8_t *x,
                          GammaSearchCondition *condition, const idx_t *idx,
                          const int32_t *coarse_dis, int32_t *distances,
                          idx_t *labels, int *total, bool store_pairs,
                          const faiss::IVFSearchParameters *params = nullptr);

  virtual GammaBinaryInvertedListScanner *get_GammaInvertedListScanner(
      bool store_pairs = false) const;

  int indexed_vec_count_;
  realtime::RTInvertIndex *rt_invert_index_ptr_;

#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_count_;
  int add_count_;
#endif
};
}  // namespace tig_gamma