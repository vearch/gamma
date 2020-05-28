/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef GAMMA_COMMON_DATA_H_
#define GAMMA_COMMON_DATA_H_

#include "field_range_index.h"
#include "gamma_api.h"
#include "log.h"
#include "online_logger.h"
#include "profile.h"
#include "utils.h"

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

enum VectorStorageType { Mmap, RocksDB };
enum RetrievalModel { IVFPQ, GPU_IVFPQ, BINARYIVF, HNSW, FLAT };

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

struct GammaSearchCondition {
  GammaSearchCondition() {
    range_query_result = nullptr;
    topn = 0;
    has_rank = false;
    multi_vector_rank = false;
    metric_type = InnerProduct;
    sort_by_docid = false;
    min_dist = -1;
    max_dist = -1;
    recall_num = 0;
    parallel_mode = 1;  // default to parallelize over inverted list
    use_direct_search = false;
    l2_sqrt = false;
    nprobe = 20;
    ivf_flat = false;

#ifdef BUILD_GPU
    range_filters = nullptr;
    range_filters_num = 0;
    term_filters = nullptr;
    term_filters_num = 0;
    profile = nullptr;
#endif  // BUILD_GPU

#ifdef PERFORMANCE_TESTING
    start_time = utils::getmillisecs();
    cur_time = start_time;
#endif
  }

  GammaSearchCondition(GammaSearchCondition *condition) {
    range_query_result = condition->range_query_result;
    topn = condition->topn;
    has_rank = condition->has_rank;
    multi_vector_rank = condition->multi_vector_rank;
    metric_type = condition->metric_type;
    sort_by_docid = condition->sort_by_docid;
    min_dist = condition->min_dist;
    max_dist = condition->max_dist;
    recall_num = condition->recall_num;
    parallel_mode = condition->parallel_mode;
    use_direct_search = condition->use_direct_search;
    l2_sqrt = condition->l2_sqrt;
    nprobe = condition->nprobe;
    ivf_flat = condition->ivf_flat;

#ifdef BUILD_GPU
    range_filters = condition->range_filters;
    range_filters_num = condition->range_filters_num;
    term_filters = condition->term_filters;
    term_filters_num = condition->term_filters_num;
    profile = condition->profile;
#endif  // BUILD_GPU
  }

  ~GammaSearchCondition() {
    range_query_result = nullptr;  // should not delete

#ifdef BUILD_GPU
    range_filters = nullptr;  // should not delete
    term_filters = nullptr;   // should not delete
    profile = nullptr;        // should not delete
#endif                        // BUILD_GPU
  }

  MultiRangeQueryResults *range_query_result;

#ifdef BUILD_GPU
  RangeFilter **range_filters;
  int range_filters_num;

  TermFilter **term_filters;
  int term_filters_num;

  Profile *profile;
#endif  // BUILD_GPU

  int topn;
  bool has_rank;
  bool multi_vector_rank;
  bool parallel_based_on_query;
  DistanceMetricType metric_type;
  bool sort_by_docid;
  float min_dist;
  float max_dist;
  int recall_num;
  int parallel_mode;
  bool use_direct_search;
  bool l2_sqrt;
  int nprobe;
  bool ivf_flat;

#ifdef PERFORMANCE_TESTING
  double cur_time;
  double start_time;
  std::stringstream perf_ss;

  void Perf(std::string &msg) {
    double old_time = cur_time;
    cur_time = utils::getmillisecs();
    perf_ss << msg << " cost [" << cur_time - old_time << "]ms ";
  }

  void Perf(const char *msg) {
    double old_time = cur_time;
    cur_time = utils::getmillisecs();
    perf_ss << msg << " cost [" << cur_time - old_time << "]ms ";
  }

  const std::stringstream &OutputPerf() {
    cur_time = utils::getmillisecs();
    perf_ss << " total cost [" << cur_time - start_time << "]ms ";
    return perf_ss;
  }

#endif
};

struct GammaQuery {
  GammaQuery() {
    vec_query = nullptr;
    vec_num = 0;
    condition = nullptr;
    logger = nullptr;
  }

  ~GammaQuery() {}
  VectorQuery **vec_query;
  int vec_num;
  GammaSearchCondition *condition;
  utils::OnlineLogger *logger;
};

struct GammaBinaryQuery {
  GammaBinaryQuery() {
    vec_query = nullptr;
    vec_num = 0;
    condition = nullptr;
    logger = nullptr;
  }

  ~GammaBinaryQuery() {}

  int *vec_id;  // binary vector id
  std::vector<int> start_pos;
  std::vector<int> sequence_len;

  ByteArray **xa;
  ByteArray **xb;
  int *d;
  int n;

  bool get_vec;

  VectorQuery **vec_query;
  int vec_num;
  GammaSearchCondition *condition;
  utils::OnlineLogger *logger;
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

struct RetrievalParams {
  DistanceMetricType metric_type;

  RetrievalParams() { metric_type = InnerProduct; }

  virtual ~RetrievalParams(){};

  bool Validate() {
    if (metric_type < InnerProduct || metric_type > L2) return false;
    return true;
  }

  virtual int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse retrieval parameters error: " << str;
      return -1;
    }

    std::string metric_type;

    if (!jp.GetString("metric_type", metric_type)) {
      if (strcasecmp("L2", metric_type.c_str()) &&
          strcasecmp("InnerProduct", metric_type.c_str())) {
        LOG(ERROR) << "invalid metric_type = " << metric_type;
        return -1;
      }
      if (!strcasecmp("L2", metric_type.c_str()))
        this->metric_type = L2;
      else
        this->metric_type = InnerProduct;
    } else {
      LOG(ERROR) << "cannot get metric type, set it when create space";
      this->metric_type = L2;
    }

    return 0;
  }

  virtual std::string ToString() {
    std::stringstream ss;
    ss << "metric_type = " << metric_type;
    return ss.str();
  }
};

struct IVFPQRetrievalParams : RetrievalParams {
  int ncentroids;     // coarse cluster center number
  int nsubvector;     // number of sub cluster center
  int nbits_per_idx;  // bit number of sub cluster center

  IVFPQRetrievalParams() : RetrievalParams() {
    ncentroids = 256;
    nsubvector = 64;
    nbits_per_idx = 8;
  }

  int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse IVFPQ retrieval parameters error: " << str;
      return -1;
    }

    std::string metric_type;

    if (!jp.GetString("metric_type", metric_type)) {
      if (strcasecmp("L2", metric_type.c_str()) &&
          strcasecmp("InnerProduct", metric_type.c_str())) {
        LOG(ERROR) << "invalid metric_type = " << metric_type;
        return -1;
      }
      if (!strcasecmp("L2", metric_type.c_str()))
        this->metric_type = L2;
      else
        this->metric_type = InnerProduct;
    } else {
      LOG(ERROR) << "cannot get metric type, set it when create space";
      return -1;
    }

    int ncentroids;
    int nsubvector;
    int nbits_per_idx;

    // -1 as default
    if (!jp.GetInt("ncentroids", ncentroids)) {
      if (ncentroids < -1) {
        LOG(ERROR) << "invalid ncentroids =" << ncentroids;
        return -1;
      }
      if (ncentroids > 0) this->ncentroids = ncentroids;
    } else {
      LOG(ERROR) << "cannot get ncentroids for ivfpq, set it when create space";
      return -1;
    }

    if (!jp.GetInt("nsubvector", nsubvector)) {
      if (nsubvector < -1) {
        LOG(ERROR) << "invalid nsubvector =" << nsubvector;
        return -1;
      }
      if (nsubvector > 0) this->nsubvector = nsubvector;
    } else {
      LOG(ERROR) << "cannot get nsubvector for ivfpq, set it when create space";
      return -1;
    }

    if (!jp.GetInt("nbits_per_idx", nbits_per_idx)) {
      if (nbits_per_idx < -1) {
        LOG(ERROR) << "invalid nbits_per_idx =" << nbits_per_idx;
        return -1;
      }
      if (nbits_per_idx > 0) this->nbits_per_idx = nbits_per_idx;
    }
    if(!Validate())
      return -1;
    return 0;
  }

  bool Validate() {
    if (metric_type < InnerProduct || metric_type > L2) return false;
    if (ncentroids <= 0 || nsubvector <= 0 || nbits_per_idx <= 0)
      return false;
    if (nsubvector % 4 != 0) {
      LOG(ERROR) << "only support multiple of 4 now, nsubvector=" << nsubvector;
      return false;
    }
    if (nbits_per_idx != 8) {
      LOG(ERROR) << "only support 8 now, nbits_per_idx=" << nbits_per_idx;
      return false;
    }
    return true;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "metric_type = " << metric_type << ", ";
    ss << "ncentroids =" << ncentroids << ", ";
    ss << "nsubvector =" << nsubvector << ", ";
    ss << "nbits_per_idx =" << nbits_per_idx;
    return ss.str();
  }
};

struct BinaryRetrievalParams : RetrievalParams {
  int nprobe;      // scan nprobe
  int ncentroids;  // coarse cluster center number

  BinaryRetrievalParams() : RetrievalParams() {
    nprobe = 20;
    ncentroids = 256;
  }

  int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse IVF retrieval parameters error: " << str;
      return -1;
    }

    int nprobe;
    int ncentroids;

    if (!jp.GetInt("nprobe", nprobe)) {
      if (nprobe < -1) {
        LOG(ERROR) << "invalid nprobe =" << nprobe;
        return -1;
      }
      if (nprobe > 0) this->nprobe = nprobe;
    } else {
      LOG(ERROR) << "cannot get nprobe for ivf, set it when create space";
      return -1;
    }

    // -1 as default
    if (!jp.GetInt("ncentroids", ncentroids)) {
      if (ncentroids < -1) {
        LOG(ERROR) << "invalid ncentroids =" << ncentroids;
        return -1;
      }
      if (ncentroids > 0) this->ncentroids = ncentroids;
    } else {
      LOG(ERROR) << "cannot get ncentroids for ivf, set it when create space";
      return -1;
    }

    return 0;
  }

  bool Validate() {
    if (nprobe <= 0 || ncentroids <= 0) return false;

    if (nprobe > ncentroids) {
      LOG(ERROR) << "nprobe=" << nprobe << " > ncentroids=" << ncentroids;
      return false;
    }
    return true;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "nprobe =" << nprobe << ", ";
    ss << "ncentroids =" << ncentroids << ", ";
    return ss.str();
  }
};

struct HNSWRetrievalParams : RetrievalParams {
  int nlinks;          // link number for hnsw graph
  int efSearch;        // search parameter for searching in hnsw graph
  int efConstruction;  // construction parameter for building hnsw graph

  HNSWRetrievalParams() : RetrievalParams() {
    nlinks = 32;
    efSearch = 64;
    efConstruction = 40;
  }

  bool Validate() {
    if (metric_type < InnerProduct || metric_type > L2) return false;
    if (nlinks < 0 || efSearch < 0 || efConstruction < 0) return false;
    return true;
  }

  int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse HNSW retrieval parameters error: " << str;
      return -1;
    }
    std::string metric_type;

    if (!jp.GetString("metric_type", metric_type)) {
      if (strcasecmp("L2", metric_type.c_str()) &&
          strcasecmp("InnerProduct", metric_type.c_str())) {
        LOG(ERROR) << "invalid metric_type = " << metric_type;
        return -1;
      }
      if (!strcasecmp("L2", metric_type.c_str()))
        this->metric_type = L2;
      else
        this->metric_type = InnerProduct;
    } else {
      LOG(ERROR) << "cannot get metric_type, set it when create space";
      return -1;
    }

    int nlinks;
    int efSearch;
    int efConstruction;

    // for -1, set as default
    if (!jp.GetInt("nlinks", nlinks)) {
      if (nlinks < -1) {
        LOG(ERROR) << "invalid nlinks = " << nlinks;
        return -1;
      }
      if (nlinks > 0) this->nlinks = nlinks;
    } else {
      LOG(ERROR) << "cannot get nlinks for hnsw, set it when create space";
      return -1;
    }

    if (!jp.GetInt("efSearch", efSearch)) {
      if (efSearch < -1) {
        LOG(ERROR) << "invalid efSearch = " << efSearch;
        return -1;
      }
      if (efSearch > 0) this->efSearch = efSearch;
    } else {
      LOG(ERROR) << "cannot get efSearch for hnsw, set it when create space";
      return -1;
    }

    if (!jp.GetInt("efConstruction", efConstruction)) {
      if (efConstruction < -1) {
        LOG(ERROR) << "invalid efConstruction = " << efConstruction;
        return -1;
      }
      if (efConstruction > 0) this->efConstruction = efConstruction;
    } else {
      LOG(ERROR)
          << "cannot get efConstruction for hnsw, set it when create space";
      return -1;
    }

    return 0;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "metric_type = " << metric_type << ", ";
    ss << "nlinks =" << nlinks << ", ";
    ss << "efSearch =" << efSearch << ", ";
    ss << "efConstruction =" << efConstruction;
    return ss.str();
  }
};

struct GammaCounters {
  int *max_docid;
  std::atomic<int> *delete_num;

  GammaCounters() {
    max_docid = nullptr;
    delete_num = nullptr;
  }

  GammaCounters(int *max_docid, std::atomic<int> *delete_num) {
    this->max_docid = max_docid;
    this->delete_num = delete_num;
  }
};

}  // namespace tig_gamma

#endif
