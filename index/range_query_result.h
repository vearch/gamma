/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef SRC_SEARCHER_INDEX_RANGE_QUERY_RESULT_H_
#define SRC_SEARCHER_INDEX_RANGE_QUERY_RESULT_H_

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "bitmap.h"
#include "log.h"

namespace tig_gamma {

// do intersection immediately
class RangeQueryResult {
 public:
  RangeQueryResult() {
    bitmap_ = nullptr;
    Clear();
  }

  ~RangeQueryResult() {
    if (bitmap_ != nullptr) {
      free(bitmap_);
      bitmap_ = nullptr;
    }
  }

  bool Has(int doc) const {
    if (doc < min_ || doc > max_) {
      return false;
    }
    doc -= min_aligned_;
    return bitmap::test(bitmap_, doc);
  }

  /**
   * @return docID in order, -1 for the end
   */
  int Next() const {
    next_++;

    int size = max_aligned_ - min_aligned_ + 1;
    while (next_ < size && not bitmap::test(bitmap_, next_)) {
      next_++;
    }
    if (next_ >= size) {
      return -1;
    }

    int doc = next_ + min_aligned_;
    return doc;
  }

  /**
   * @return size of docID list
   */
  int Size() const { return n_doc_; }

  void Clear() {
    min_ = std::numeric_limits<int>::max();
    max_ = 0;
    next_ = -1;
    n_doc_ = -1;
    if (bitmap_ != nullptr) {
      free(bitmap_);
      bitmap_ = nullptr;
    }
  }

  void SetRange(int x, int y) {
    min_ = std::min(min_, x);
    max_ = std::max(max_, y);
    min_aligned_ = (min_ / 8) * 8;
    max_aligned_ = (max_ / 8 + 1) * 8 - 1;
  }

  void Resize() {
    int n = max_aligned_ - min_aligned_ + 1;
    assert(n > 0);
    if (bitmap_ != nullptr) {
      free(bitmap_);
      bitmap_ = nullptr;
    }

    int bytes_count = -1;
    if (bitmap::create(bitmap_, bytes_count, n) != 0) {
      LOG(ERROR) << "Cannot create bitmap!";
      return;
    }
  }

  void Set(int pos) { bitmap::set(bitmap_, pos); }

  int Min() const { return min_; }
  int Max() const { return max_; }

  int MinAligned() { return min_aligned_; }
  int MaxAligned() { return max_aligned_; }

  char *&Ref() { return bitmap_; }

  void SetDocNum(int num) { n_doc_ = num; }

  /**
   * @return sorted docIDs
   */
  std::vector<int> ToDocs() const;  // WARNING: build dynamically
  void Output();

 private:
  int min_;
  int max_;
  int min_aligned_;
  int max_aligned_;

  mutable int next_;
  mutable int n_doc_;

  char *bitmap_;
};

// do intersection lazily
class MultiRangeQueryResults {
 public:
  MultiRangeQueryResults() { Clear(); }

  ~MultiRangeQueryResults() {
    delete all_results_;
    all_results_ = nullptr;
  }

  // Take full advantage of multi-core while recalling
  bool Has(int doc) const {
    return all_results_->Has(doc);
  }

  void Clear() {
    min_ = 0;
    max_ = std::numeric_limits<int>::max();
    all_results_ = nullptr;
  }

 public:
  void Add(RangeQueryResult *r) {
    all_results_ = r;

    // the maximum of the minimum(s)
    if (r->Min() > min_) {
      min_ = r->Min();
    }
    // the minimum of the maximum(s)
    if (r->Max() < max_) {
      max_ = r->Max();
    }
  }

  int Min() const { return min_; }
  int Max() const { return max_; }

  /** WARNING: build dynamically
   * @return sorted docIDs
   */
  std::vector<int> ToDocs() const;

  const RangeQueryResult *GetAllResult() const {
    return all_results_;
  }

 private:
  int min_;
  int max_;

  RangeQueryResult *all_results_;
};

}  // namespace tig_gamma

#endif  // SRC_SEARCHER_INDEX_RANGE_QUERY_RESULT_H_
