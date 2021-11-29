/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include "common/common_query_data.h"
#include "index/retrieval_model.h"
#include "table/field_range_index.h"
#include "table/table.h"
#include "util/log.h"
#include "util/online_logger.h"
#include "util/utils.h"
#include "vector/raw_vector.h"

namespace tig_gamma {

const std::string EXTRA_VECTOR_FIELD_SOURCE = "source";
const std::string EXTRA_VECTOR_FIELD_SCORE = "score";
const std::string EXTRA_VECTOR_FIELD_NAME = "field";
const std::string EXTRA_VECTOR_RESULT = "vector_result";

const float GAMMA_INDEX_RECALL_RATIO = 1.0f;

enum class ResultCode : std::uint16_t {
#define DefineResultCode(Name, Value) Name = Value,
#include "definition_list.h"
#undef DefineResultCode
  Undefined
};

enum class VectorStorageType : std::uint8_t { MemoryOnly, Mmap, RocksDB };

struct VectorDocField {
  std::string name;
  double score;
  char *source;
  int source_len;
};

struct VectorDoc {
  VectorDoc() {
    docid = -1;
    score = 0.0f;
  }

  ~VectorDoc() {
    if (fields) {
      delete[] fields;
      fields = nullptr;
    }
  }

  bool init(std::string *vec_names, int vec_num) {
    if (vec_num <= 0) {
      fields = nullptr;
      fields_len = 0;
      return true;
    }
    fields = new (std::nothrow) VectorDocField[vec_num];
    if (fields == nullptr) {
      return false;
    }
    for (int i = 0; i < vec_num; i++) {
      fields[i].name = vec_names[i];
    }
    fields_len = vec_num;
    return true;
  }

  int docid;
  double score;
  struct VectorDocField *fields;
  int fields_len;
};

class GammaSearchCondition : public RetrievalContext {
 public:
  GammaSearchCondition() {
    range_query_result = nullptr;
    topn = 0;
    multi_vector_rank = false;
    metric_type = DistanceComputeType::INNER_PRODUCT;
    sort_by_docid = false;
    brute_force_search = false;
    l2_sqrt = false;
    has_rank = 1;
    min_score = std::numeric_limits<float>::min();
    max_score = std::numeric_limits<float>::max();

#ifdef BUILD_GPU
    table = nullptr;
#endif  // BUILD_GPU
  }

  GammaSearchCondition(GammaSearchCondition *condition) {
    range_query_result = condition->range_query_result;
    topn = condition->topn;
    multi_vector_rank = condition->multi_vector_rank;
    metric_type = condition->metric_type;
    sort_by_docid = condition->sort_by_docid;
    brute_force_search = condition->brute_force_search;
    l2_sqrt = condition->l2_sqrt;
    has_rank = condition->has_rank;

#ifdef BUILD_GPU
    range_filters = condition->range_filters;
    term_filters = condition->term_filters;
    table = condition->table;
#endif  // BUILD_GPU
  }

  ~GammaSearchCondition() {
    range_query_result = nullptr;  // should not delete

#ifdef BUILD_GPU
    table = nullptr;  // should not delete
#endif                // BUILD_GPU
  }

  MultiRangeQueryResults *range_query_result;

#ifdef BUILD_GPU
  std::vector<struct RangeFilter> range_filters;
  std::vector<struct TermFilter> term_filters;

  Table *table;
#endif  // BUILD_GPU

  int topn;
  bool multi_vector_rank;
  enum DistanceComputeType metric_type;
  bool sort_by_docid;
  bool brute_force_search;
  bool l2_sqrt;
  bool has_rank;
  std::string retrieval_parameters;
  float min_score;
  float max_score;

  bool IsSimilarScoreValid(float score) const override {
    return (score <= max_score) && (score >= min_score);
  };

  bool IsValid(int id) const override {
    int docid = raw_vec->VidMgr()->VID2DocID(id);
    if ((range_query_result != nullptr && not range_query_result->Has(docid)) ||
        docids_bitmap->Test(docid) == true) {
      return false;
    }
    return true;
  };

  void Init(float min_score, float max_score,
            bitmap::BitmapManager *docids_bitmap, RawVector *raw_vec) {
    this->min_score = min_score;
    this->max_score = max_score;
    this->docids_bitmap = docids_bitmap;
    this->raw_vec = raw_vec;
  }

  int VID2DocID(int vid) { return raw_vec->VidMgr()->VID2DocID(vid); }
  MultiRangeQueryResults *RangeQueryResult() { return range_query_result; }

 private:
  bitmap::BitmapManager *docids_bitmap;
  const RawVector *raw_vec;
};

struct GammaQuery {
  GammaQuery() { condition = nullptr; }

  ~GammaQuery() {
    if (condition) {
      delete condition;
      condition = nullptr;
    }
  }

  std::vector<struct VectorQuery> vec_query;
  GammaSearchCondition *condition;
};

struct GammaResult {
  GammaResult() {
    topn = 0;
    total = 0;
    results_count = 0;
    docs = nullptr;
  }
  ~GammaResult() {
    if (docs) {
      for (int i = 0; i < topn; i++) {
        if (docs[i]) {
          delete docs[i];
          docs[i] = nullptr;
        }
      }
      delete[] docs;
      docs = nullptr;
    }
  }

  bool init(int n, std::string *vec_names, int vec_num) {
    topn = n;
    docs = new (std::nothrow) VectorDoc *[topn];
    if (!docs) {
      // LOG(ERROR) << "docs in CommonDocs init error!";
      return false;
    }
    for (int i = 0; i < n; i++) {
      docs[i] = new VectorDoc();
      if (!docs[i]->init(vec_names, vec_num)) {
        return false;
      }
    }
    return true;
  }

  int topn;
  int total;
  int results_count;

  VectorDoc **docs;
};

}  // namespace tig_gamma
