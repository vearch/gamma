/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This faiss source code is licensed under the MIT license.
 * https://github.com/facebookresearch/faiss/blob/master/LICENSE
 *
 *
 * The works below are modified based on faiss:
 * 1. Replace the static batch indexing with real time indexing
 * 2. Add the fine-grained sort after PQ coarse sort
 * 3. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2019 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#include "gamma_index_ivfpq.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "bitmap.h"
#include "error_code.h"
#include "faiss/IndexFlat.h"
#include "gamma_index_io.h"
#include "mmap_raw_vector.h"
#include "omp.h"
#include "utils.h"

namespace tig_gamma {

static inline void ConvertVectorDim(size_t num, int raw_d, int d,
                                    const float *raw_vec, float *vec) {
  memset(vec, 0, num * d * sizeof(float));

#pragma omp parallel for
  for (size_t i = 0; i < num; ++i) {
    for (int j = 0; j < raw_d; ++j) {
      vec[i * d + j] = raw_vec[i * raw_d + j];
    }
  }
}

IndexIVFPQStats indexIVFPQ_stats;

struct IVFPQModelParams {
  int ncentroids;     // coarse cluster center number
  int nsubvector;     // number of sub cluster center
  int nbits_per_idx;  // bit number of sub cluster center
  DistanceComputeType metric_type;
  bool has_hnsw;
  int nlinks;          // link number for hnsw graph
  int efConstruction;  // construction parameter for building hnsw graph
  int efSearch;        // search parameter for search in hnsw graph
  bool has_opq;
  int opq_nsubvector;  // number of sub cluster center of opq
  int bucket_init_size; // original size of RTInvertIndex bucket
  int bucket_max_size; // max size of RTInvertIndex bucket

  IVFPQModelParams() {
    ncentroids = 2048;
    nsubvector = 64;
    nbits_per_idx = 8;
    metric_type = DistanceComputeType::INNER_PRODUCT;
    has_hnsw = false;
    nlinks = 32;
    efConstruction = 200;
    efSearch = 64;
    has_opq = false;
    opq_nsubvector = 64;
    bucket_init_size = 1000;
    bucket_max_size = 1280000;
  }

  int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse IVFPQ retrieval parameters error: " << str;
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

    int bucket_init_size;
    int bucket_max_size;

    // -1 as default
    if (!jp.GetInt("bucket_init_size", bucket_init_size)) {
      if (bucket_init_size < -1) {
        LOG(ERROR) << "invalid bucket_init_size =" << bucket_init_size;
        return -1;
      }
      if (bucket_init_size > 0) this->bucket_init_size = bucket_init_size;
    }

    if (!jp.GetInt("bucket_max_size", bucket_max_size)) {
      if (bucket_max_size < -1) {
        LOG(ERROR) << "invalid bucket_max_size =" << bucket_max_size;
        return -1;
      }
      if (bucket_max_size > 0) this->bucket_max_size = bucket_max_size;
    }

    std::string metric_type;

    if (!jp.GetString("metric_type", metric_type)) {
      if (strcasecmp("L2", metric_type.c_str()) &&
          strcasecmp("InnerProduct", metric_type.c_str())) {
        LOG(ERROR) << "invalid metric_type = " << metric_type;
        return -1;
      }
      if (!strcasecmp("L2", metric_type.c_str()))
        this->metric_type = DistanceComputeType::L2;
      else
        this->metric_type = DistanceComputeType::INNER_PRODUCT;
    }

    utils::JsonParser jp_hnsw;
    if (!jp.GetObject("hnsw", jp_hnsw)) {
      has_hnsw = true;
      int nlinks;
      int efConstruction;
      int efSearch;
      // -1 as default
      if (!jp_hnsw.GetInt("nlinks", nlinks)) {
        if (nlinks < -1) {
          LOG(ERROR) << "invalid nlinks = " << nlinks;
          return -1;
        }
        if(nlinks > 0) this->nlinks = nlinks;
      }

      if (!jp_hnsw.GetInt("efConstruction", efConstruction)) {
        if (efConstruction < -1) {
          LOG(ERROR) << "invalid efConstruction = " << efConstruction;
          return -1;
        }
        if(efConstruction > 0) this->efConstruction = efConstruction;
      }

      if (!jp_hnsw.GetInt("efSearch", efSearch)) {
        if (efSearch < -1) {
          LOG(ERROR) << "invalid efSearch = " << efSearch;
          return -1;
        }
        if(efSearch > 0) this->efSearch = efSearch;
      }
    }

    utils::JsonParser jp_opq;
    if (!jp.GetObject("opq", jp_opq)) {
      has_opq = true;
      int opq_nsubvector;
      // -1 as default
      if (!jp_opq.GetInt("nsubvector", opq_nsubvector)) {
        if (nsubvector < -1) {
          LOG(ERROR) << "invalid opq_nsubvector = " << opq_nsubvector;
          return -1;
        }
        if (opq_nsubvector > 0) this->opq_nsubvector = opq_nsubvector;
      } 
    }

    if (!Validate()) return -1;
    return 0;
  }

  bool Validate() {
    if (ncentroids <= 0 || nsubvector <= 0 || nbits_per_idx <= 0) return false;
    if (nbits_per_idx != 8) {
      LOG(ERROR) << "only support 8 now, nbits_per_idx=" << nbits_per_idx;
      return false;
    }

    return true;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "ncentroids =" << ncentroids << ", ";
    ss << "nsubvector =" << nsubvector << ", ";
    ss << "nbits_per_idx =" << nbits_per_idx << ", ";
    ss << "metric_type =" << (int)metric_type << ", ";
    ss << "bucket_init_size =" << bucket_init_size << ", ";
    ss << "bucket_max_size =" << bucket_max_size;

    if (has_hnsw) {
      ss << ", hnsw: nlinks=" << nlinks << ", ";
      ss << "efConstrction=" << efConstruction << ", ";
      ss << "efSearch=" << efSearch;
    }
    if (has_opq) {
      ss << ", opq: nsubvector=" << opq_nsubvector;
    }

    return ss.str();
  }

  int ToJson(utils::JsonParser &jp) { return 0; }
};

REGISTER_MODEL(IVFPQ, GammaIVFPQIndex)

GammaIVFPQIndex::GammaIVFPQIndex() : indexed_vec_count_(0) {
  compaction_ = false;
  compact_bucket_no_ = 0;
  compacted_num_ = 0;
  updated_num_ = 0;
  is_trained = false;
  opq_ = nullptr;
#ifdef PERFORMANCE_TESTING
  search_count_ = 0;
  add_count_ = 0;
#endif
}

GammaIVFPQIndex::~GammaIVFPQIndex() {
  if (rt_invert_index_ptr_) {
    delete rt_invert_index_ptr_;
    rt_invert_index_ptr_ = nullptr;
  }
  if (invlists) {
    delete invlists;
    invlists = nullptr;
  }
  if (quantizer) {
    delete quantizer;  // it will not be delete in parent class
    quantizer = nullptr;
  }
  if (opq_) {
    delete opq_;
    opq_ = nullptr;
  }

  CHECK_DELETE(model_param_);
}

faiss::InvertedListScanner *GammaIVFPQIndex::get_InvertedListScanner(
    bool store_pairs, faiss::MetricType metric_type) {
  return GetGammaInvertedListScanner(store_pairs, metric_type);
}

GammaInvertedListScanner *GammaIVFPQIndex::GetGammaIVFFlatScanner(
    size_t d, faiss::MetricType metric_type) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    auto scanner = new GammaIVFFlatScanner<faiss::METRIC_INNER_PRODUCT,
                                           faiss::CMin<float, int64_t>>(d);
    return scanner;
  } else if (metric_type == faiss::METRIC_L2) {
    auto scanner =
        new GammaIVFFlatScanner<faiss::METRIC_L2, faiss::CMax<float, int64_t>>(
            d);
    return scanner;
  } else {
    LOG(ERROR) << "metric type not supported";
  }
  return nullptr;
}

GammaInvertedListScanner *GammaIVFPQIndex::GetGammaInvertedListScanner(
    bool store_pairs, faiss::MetricType metric_type) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    auto scanner =
        new GammaIVFPQScanner<faiss::METRIC_INNER_PRODUCT,
                              faiss::CMin<float, idx_t>, 2>(*this, store_pairs);
    return scanner;
  } else if (metric_type == faiss::METRIC_L2) {
    auto scanner =
        new GammaIVFPQScanner<faiss::METRIC_L2, faiss::CMax<float, idx_t>, 2>(
            *this, store_pairs);
    return scanner;
  }
  return nullptr;
}

int GammaIVFPQIndex::Init(const std::string &model_parameters) {
  model_param_ = new IVFPQModelParams();
  IVFPQModelParams &ivfpq_param = *model_param_;
  if (model_parameters != "" && ivfpq_param.Parse(model_parameters.c_str())) {
    return -1;
  }
  LOG(INFO) << ivfpq_param.ToString();

  d = vector_->MetaInfo()->Dimension();

  if (d % ivfpq_param.nsubvector != 0) {
    d = (d / ivfpq_param.nsubvector + 1) * ivfpq_param.nsubvector;
    LOG(INFO) << "Dimension [" << vector_->MetaInfo()->Dimension()
              << "] cannot divide by nsubvector [" << ivfpq_param.nsubvector
              << "], adjusted to [" << d << "]";
  }

  RawVector *raw_vec = dynamic_cast<RawVector *>(vector_);

  nlist = ivfpq_param.ncentroids;
  if (ivfpq_param.has_hnsw == false) {
    quantizer = new faiss::IndexFlatL2(d);
    quantizer_type_ = 0;
  } else {
    faiss::IndexHNSWFlat *hnsw_flat = new faiss::IndexHNSWFlat(d, ivfpq_param.nlinks);
    hnsw_flat->hnsw.efSearch = ivfpq_param.efSearch;
    hnsw_flat->hnsw.efConstruction = ivfpq_param.efConstruction;
    hnsw_flat->hnsw.search_bounded_queue = false;
    quantizer = hnsw_flat;
    quantizer_type_ = 1;
  }

  if (ivfpq_param.has_opq) {
    if (d % ivfpq_param.opq_nsubvector != 0) {
      LOG(ERROR) << d << " % " << ivfpq_param.opq_nsubvector 
                 << " != 0, opq nsubvector should be divisible by dimension.";
      return -2; 
    }
    opq_ = new faiss::OPQMatrix(d, ivfpq_param.opq_nsubvector, d);
  }

  pq.d = d;
  pq.M = ivfpq_param.nsubvector;
  pq.nbits = ivfpq_param.nbits_per_idx;
  pq.set_derived_values();

  own_fields = false;
  quantizer_trains_alone = 0;
  clustering_index = nullptr;
  cp.niter = 10;

  code_size = pq.code_size;
  is_trained = false;
  by_residual = true;
  use_precomputed_table = 0;
  scan_table_threshold = 0;

  polysemous_training = nullptr;
  do_polysemous_training = false;
  polysemous_ht = 0;

  // if nlist is very large, 
  // the size of RTInvertIndex bucket should be smaller
  rt_invert_index_ptr_ = new realtime::RTInvertIndex(
    this->nlist, this->code_size, raw_vec->VidMgr(), raw_vec->Bitmap(), 
    ivfpq_param.bucket_init_size, ivfpq_param.bucket_max_size);

  if (this->invlists) {
    delete this->invlists;
    this->invlists = nullptr;
  }
  d_ = d;
  bool ret = rt_invert_index_ptr_->Init();

  if (ret) {
    this->invlists =
        new realtime::RTInvertedLists(rt_invert_index_ptr_, nlist, code_size);
  }

  metric_type_ = ivfpq_param.metric_type;
  if (metric_type_ == DistanceComputeType::INNER_PRODUCT) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }

  // default value, nprobe will be passed at search time
  this->nprobe = 80;
  return 0;
}

RetrievalParameters *GammaIVFPQIndex::Parse(const std::string &parameters) {
  if (parameters == "") {
    return new IVFPQRetrievalParameters(metric_type_);
  }

  utils::JsonParser jp;
  if (jp.Parse(parameters.c_str())) {
    LOG(ERROR) << "parse retrieval parameters error: " << parameters;
    return nullptr;
  }

  std::string metric_type;
  IVFPQRetrievalParameters *retrieval_params = new IVFPQRetrievalParameters();
  if (!jp.GetString("metric_type", metric_type)) {
    if (strcasecmp("L2", metric_type.c_str()) &&
        strcasecmp("InnerProduct", metric_type.c_str())) {
      LOG(ERROR) << "invalid metric_type = " << metric_type
                 << ", so use default value.";
    }
    if (!strcasecmp("L2", metric_type.c_str())) {
      retrieval_params->SetDistanceComputeType(DistanceComputeType::L2);
    } else {
      retrieval_params->SetDistanceComputeType(
          DistanceComputeType::INNER_PRODUCT);
    }
  } else {
    retrieval_params->SetDistanceComputeType(metric_type_);
  }

  int recall_num;
  int nprobe;
  int parallel_on_queries;
  int ivf_flat;

  if (!jp.GetInt("recall_num", recall_num)) {
    if (recall_num > 0) {
      retrieval_params->SetRecallNum(recall_num);
    }
  }

  if (!jp.GetInt("nprobe", nprobe)) {
    if (nprobe > 0) {
      retrieval_params->SetNprobe(nprobe);
    }
  }

  if (!jp.GetInt("parallel_on_queries", parallel_on_queries)) {
    if (parallel_on_queries != 0) {
      retrieval_params->SetParallelOnQueries(true);
    } else {
      retrieval_params->SetParallelOnQueries(false);
    }
  }

  if (!jp.GetInt("ivf_flat", ivf_flat)) {
    if (ivf_flat != 0) {
      retrieval_params->SetIvfFlat(true);
    }
  }
  return retrieval_params;
}

int GammaIVFPQIndex::Indexing() {
  if (this->is_trained) {
    LOG(INFO) << "gamma ivfpq index is already trained, skip indexing";
    return 0;
  }
  RawVector *raw_vec = dynamic_cast<RawVector *>(vector_);
  size_t vectors_count = raw_vec->MetaInfo()->Size();
  size_t num;
  if (quantizer_type_ == 0) {
    if (vectors_count < 8192) {
        LOG(ERROR) << "vector total count [" << vectors_count
                   << "] less then 8192, failed!";
      return -1;
    }
    num = vectors_count > 100000 ? 100000 : vectors_count;
  } else {
    // for this case, can use more data for training
    if (vectors_count < nlist) {
      LOG(ERROR) << "vector total count [" << vectors_count
               << "] less then " << nlist << ", failed!";
      return -2;
    }
    num = vectors_count > nlist * 256 ? nlist * 256 : vectors_count;
  }

  ScopeVectors headers;
  std::vector<int> lens;
  raw_vec->GetVectorHeader(0, num, headers, lens);

  // merge vectors
  int raw_d = raw_vec->MetaInfo()->Dimension();
  const uint8_t *train_raw_vec = nullptr;
  utils::ScopeDeleter1<uint8_t> del_train_raw_vec;
  if (lens.size() == 1) {
    train_raw_vec = headers.Get(0);
  } else {
    train_raw_vec = new uint8_t[raw_d * num * sizeof(float)];
    del_train_raw_vec.set(train_raw_vec);
    size_t offset = 0;
    for (size_t i = 0; i < headers.Size(); ++i) {
      memcpy((void *)(train_raw_vec + offset), (void *)headers.Get(i),
             sizeof(float) * raw_d * lens[i]);
      offset += sizeof(float) * raw_d * lens[i];
    }
  }

  const float *train_vec = nullptr;

  if (d_ > raw_d) {
    float *vec = new float[num * d_];

    ConvertVectorDim(num, raw_d, d, (const float *)train_raw_vec, vec);

    train_vec = vec;
  } else {
    train_vec = (const float *)train_raw_vec;
  }
  
  const float *xt = nullptr;
  utils::ScopeDeleter1<float> del_xt;
  if (opq_ != nullptr) {
    opq_->train(num, train_vec);
    xt = opq_->apply(num, train_vec);
    del_xt.set(xt == train_vec ? nullptr : xt);
  } else {
    xt = train_vec;
  }

  train(num, xt);

  if (d_ > raw_d) {
    delete train_vec;
  }

  LOG(INFO) << "train successed!";
  return 0;
}

static float *compute_residuals(const faiss::Index *quantizer, long n,
                                const float *x, const idx_t *list_nos) {
  size_t d = quantizer->d;
  float *residuals = new float[n * d];
  for (int i = 0; i < n; i++) {
    if (list_nos[i] < 0)
      memset(residuals + i * d, 0, sizeof(*residuals) * d);
    else
      quantizer->compute_residual(x + i * d, residuals + i * d, list_nos[i]);
  }
  return residuals;
}

int GammaIVFPQIndex::Delete(const std::vector<int64_t> &ids) {
  std::vector<int> vids(ids.begin(), ids.end());
  rt_invert_index_ptr_->Delete(vids.data(), ids.size());
  return 0;
}

int GammaIVFPQIndex::Update(const std::vector<int64_t> &ids,
                            const std::vector<const uint8_t *> &vecs) {
  int raw_d = vector_->MetaInfo()->Dimension();
  for (size_t i = 0; i < ids.size(); i++) {
    const float *vec = nullptr;
    utils::ScopeDeleter1<float> del_vec;
    const float *add_vec = reinterpret_cast<const float *>(vecs[i]);
    if (d_ > raw_d) {
      float *extend_vec = new float[d_];
      ConvertVectorDim(1, raw_d, d_, add_vec, extend_vec);
      vec = (const float *)extend_vec;
      del_vec.set(vec);
    } else {
      vec = add_vec;
    }
    const float *applied_vec = nullptr;
    utils::ScopeDeleter1<float> del_applied;
    if (opq_ != nullptr) {
      applied_vec = opq_->apply(1, vec);
      del_applied.set(applied_vec == vec ? nullptr : applied_vec);
    } else {
      applied_vec = vec;
    }
    idx_t idx = -1;
    quantizer->assign(1, applied_vec, &idx);

    std::vector<uint8_t> xcodes;
    xcodes.resize(code_size);
    const float *to_encode = nullptr;
    utils::ScopeDeleter1<float> del_to_encode;

    if (by_residual) {
      to_encode = compute_residuals(quantizer, 1, applied_vec, &idx);
      del_to_encode.set(to_encode);
    } else {
      to_encode = applied_vec;
    }
    pq.compute_codes(to_encode, xcodes.data(), 1);
    rt_invert_index_ptr_->Update(idx, ids[i], xcodes);
  }
  updated_num_ += ids.size();
  LOG(INFO) << "update index success! size=" << ids.size()
            << ", total=" << updated_num_;

  // now check id need to do compaction
  rt_invert_index_ptr_->CompactIfNeed();
  return 0;
}

bool GammaIVFPQIndex::Add(int n, const uint8_t *vec) {
#ifdef PERFORMANCE_TESTING
  double t0 = faiss::getmillisecs();
#endif
  std::map<int, std::vector<long>> new_keys;
  std::map<int, std::vector<uint8_t>> new_codes;

  idx_t *idx;
  utils::ScopeDeleter<idx_t> del_idx;
  const float *add_vec = reinterpret_cast<const float *>(vec);
  const float *add_vec_head = nullptr;
  utils::ScopeDeleter<float> del_vec;
  int raw_d = vector_->MetaInfo()->Dimension();
  if (d_ > raw_d) {
    float *vector = new float[n * d_];
    ConvertVectorDim(n, raw_d, d, add_vec, vector);
    add_vec_head = vector;
    del_vec.set(add_vec_head);
  } else {
    add_vec_head = add_vec;
  }

  const float *applied_vec = nullptr;
  utils::ScopeDeleter1<float> del_applied;
  if (opq_ != nullptr) {
    applied_vec = opq_->apply(n, add_vec_head);
    del_applied.set(applied_vec == add_vec_head ? nullptr : applied_vec);
  } else {
    applied_vec = add_vec_head;
  }

  idx_t *idx0 = new idx_t[n];
  quantizer->assign(n, applied_vec, idx0);
  idx = idx0;
  del_idx.set(idx);

  uint8_t *xcodes = new uint8_t[n * code_size];
  utils::ScopeDeleter<uint8_t> del_xcodes(xcodes);

  const float *to_encode = nullptr;
  utils::ScopeDeleter<float> del_to_encode;
  
  if (by_residual) {
    to_encode = compute_residuals(quantizer, n, applied_vec, idx);
    del_to_encode.set(to_encode);
  } else {
    to_encode = applied_vec;
  }
  pq.compute_codes(to_encode, xcodes, n);

  size_t n_ignore = 0;
  long vid = indexed_vec_count_;
  for (int i = 0; i < n; i++) {
    long key = idx[i];
    assert(key < (long)nlist);
    if (key < 0) {
      n_ignore++;
      LOG(WARNING) << "ivfpq add invalid key=" << key
                   << ", vid=" << vid;
      key = vid % nlist;   
    }

    // long id = (long)(indexed_vec_count_++);
    uint8_t *code = xcodes + i * code_size;

    new_keys[key].push_back(vid++);

    size_t ofs = new_codes[key].size();
    new_codes[key].resize(ofs + code_size);
    memcpy((void *)(new_codes[key].data() + ofs), (void *)code, code_size);
  }

  /* stage 2 : add invert info to invert index */
  if (!rt_invert_index_ptr_->AddKeys(new_keys, new_codes)) {
    return false;
  }
  indexed_vec_count_ = vid;
#ifdef PERFORMANCE_TESTING
  add_count_ += n;
  if (add_count_ >= 10000) {
    double t1 = faiss::getmillisecs();
    LOG(INFO) << "Add time [" << (t1 - t0) / n << "]ms, count "
              << indexed_vec_count_;
    // rt_invert_index_ptr_->PrintBucketSize();
    add_count_ = 0;
  }
#endif
  return true;
}

int GammaIVFPQIndex::Search(RetrievalContext *retrieval_context, int n,
                            const uint8_t *x, int k, float *distances,
                            idx_t *labels) {
  IVFPQRetrievalParameters *retrieval_params =
      dynamic_cast<IVFPQRetrievalParameters *>(
          retrieval_context->RetrievalParams());

  utils::ScopeDeleter1<IVFPQRetrievalParameters> del_params;
  if (retrieval_params == nullptr) {
    retrieval_params = new IVFPQRetrievalParameters();
    del_params.set(retrieval_params);
  }

  GammaSearchCondition *condition =
      dynamic_cast<GammaSearchCondition *>(retrieval_context);
  if (condition->brute_force_search == true || is_trained == false) {
    // reset retrieval_params
    delete retrieval_context->RetrievalParams();
    retrieval_context->retrieval_params_ = new FlatRetrievalParameters(
        retrieval_params->ParallelOnQueries(), retrieval_params->GetDistanceComputeType());
    int ret =
        GammaFLATIndex::Search(retrieval_context, n, x, k, distances, labels);
    return ret;
  }

  int nprobe = this->nprobe;
  if (retrieval_params->Nprobe() > 0 &&
      (size_t)retrieval_params->Nprobe() <= this->nlist) {
    nprobe = retrieval_params->Nprobe();
  } else {
    LOG(WARNING) << "Error nprobe for search, so using default value:"
                 << this->nprobe;
    retrieval_params->SetNprobe(this->nprobe);
  }

  const float *xq = reinterpret_cast<const float *>(x);
  const float *applied_xq = nullptr;
  utils::ScopeDeleter1<float> del_applied;
  if (opq_ == nullptr) {
    applied_xq = xq;
  } else {
    applied_xq = opq_->apply(n, xq);
    del_applied.set(applied_xq == xq ? nullptr : applied_xq);
  }

  std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
  std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

  if (retrieval_params->IvfFlat() == true) {
    quantizer->search(n, xq, nprobe, coarse_dis.get(), idx.get());
  } else {
    quantizer->search(n, applied_xq, nprobe, coarse_dis.get(), idx.get());
  }
  this->invlists->prefetch_lists(idx.get(), n * nprobe);

  if (retrieval_params->IvfFlat() == true) {
    // just use xq
    search_ivf_flat(retrieval_context, n, xq, k, idx.get(), coarse_dis.get(),
                    distances, labels, nprobe, false);
  } else {
    search_preassigned(retrieval_context, n, xq, applied_xq, k, idx.get(), coarse_dis.get(),
                       distances, labels, nprobe, false);
  }
  return 0;
}

namespace {

using HeapForIP = faiss::CMin<float, idx_t>;
using HeapForL2 = faiss::CMax<float, idx_t>;

// intialize + reorder a result heap

int init_result(faiss::MetricType metric_type, int k, float *simi,
                idx_t *idxi) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    faiss::heap_heapify<HeapForIP>(k, simi, idxi);
  } else {
    faiss::heap_heapify<HeapForL2>(k, simi, idxi);
  }
  return 0;
};

int reorder_result(faiss::MetricType metric_type, int k, float *simi,
                   idx_t *idxi) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    faiss::heap_reorder<HeapForIP>(k, simi, idxi);
  } else {
    faiss::heap_reorder<HeapForL2>(k, simi, idxi);
  }
  return 0;
};

// single list scan using the current scanner (with query
// set porperly) and storing results in simi and idxi
size_t scan_one_list(GammaInvertedListScanner *scanner, idx_t key,
                     float coarse_dis_i, float *simi, idx_t *idxi, int k,
                     idx_t nlist, faiss::InvertedLists *invlists,
                     bool store_pairs, bool ivf_flat,
                     MemoryRawVector *mem_raw_vec = nullptr) {
  if (key < 0) {
    // not enough centroids for multiprobe
    return 0;
  }
  if (key >= (idx_t)nlist) {
    LOG(INFO) << "Invalid key=" << key << ", nlist=" << nlist;
    return 0;
  }

  size_t list_size = invlists->list_size(key);

  // don't waste time on empty lists
  if (list_size == 0) {
    return 0;
  }

  std::unique_ptr<faiss::InvertedLists::ScopedIds> sids;
  const idx_t *ids = nullptr;

  if (!store_pairs) {
    sids.reset(new faiss::InvertedLists::ScopedIds(invlists, key));
    ids = sids->get();
  }

  scanner->set_list(key, coarse_dis_i);

  // scan_codes need uint8_t *
  const uint8_t *codes = nullptr;

  if (ivf_flat) {
    codes = reinterpret_cast<uint8_t *>(mem_raw_vec);
  } else {
    faiss::InvertedLists::ScopedCodes scodes(invlists, key);
    codes = scodes.get();
  }
  scanner->scan_codes(list_size, codes, ids, simi, idxi, k);

  return list_size;
};

void compute_dis(int k, const float *xi, float *simi, idx_t *idxi,
                 float *recall_simi, idx_t *recall_idxi, int recall_num,
                 bool has_rank, faiss::MetricType metric_type,
                 VectorReader *vec, RetrievalContext *retrieval_context) {
  if (has_rank == true) {
    ScopeVectors scope_vecs;
    std::vector<idx_t> vids(recall_idxi, recall_idxi + recall_num);
    vec->Gets(vids, scope_vecs);
    int raw_d = vec->MetaInfo()->Dimension();
    for (int j = 0; j < recall_num; j++) {
      if (recall_idxi[j] == -1) continue;
      float dis = 0;
      const float *vec = reinterpret_cast<const float *>(scope_vecs.Get(j));
      if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        dis = faiss::fvec_inner_product(xi, vec, raw_d);
      } else {
        dis = faiss::fvec_L2sqr(xi, vec, raw_d);
      }

      if (retrieval_context->IsSimilarScoreValid(dis) == true) {
        if (metric_type == faiss::METRIC_INNER_PRODUCT) {
          if (HeapForIP::cmp(simi[0], dis)) {
            faiss::heap_pop<HeapForIP>(k, simi, idxi);
            long id = recall_idxi[j];
            faiss::heap_push<HeapForIP>(k, simi, idxi, dis, id);
          }
        } else {
          if (HeapForL2::cmp(simi[0], dis)) {
            faiss::heap_pop<HeapForL2>(k, simi, idxi);
            long id = recall_idxi[j];
            faiss::heap_push<HeapForL2>(k, simi, idxi, dis, id);
          }
        }
      }
    }
    reorder_result(metric_type, k, simi, idxi);
  } else {
    // compute without rank
    int i = 0;
    reorder_result(metric_type, recall_num, recall_simi, recall_idxi);
    for (int j = 0; j < recall_num; j++) {
      if (recall_idxi[j] == -1) continue;
      float dis = recall_simi[j];

      if (retrieval_context->IsSimilarScoreValid(dis) == true) {
        simi[i] = dis;
        idxi[i] = recall_idxi[j];
        ++i;
      }
      if (i >= k) break;
    }
  };
}

}  // namespace

void GammaIVFPQIndex::search_ivf_flat(
    RetrievalContext *retrieval_context, int n, const float *x, int k,
    const idx_t *keys, const float *coarse_dis, float *distances, idx_t *labels,
    int nprobe, bool store_pairs, const faiss::IVFSearchParameters *params) {
  if (k <= 0) {
    LOG(WARNING) << "topK should greater then 0, topK = " << k;
    return;
  }

  MemoryRawVector *mem_raw_vec = dynamic_cast<MemoryRawVector *>(vector_);
  if (mem_raw_vec == nullptr) {
    LOG(ERROR) << "IVF FLAT can only work on memory raw vector";
    memset(labels, -1, n * sizeof(idx_t) * k);
    return;
  }

  IVFPQRetrievalParameters *retrieval_params =
      dynamic_cast<IVFPQRetrievalParameters *>(
          retrieval_context->RetrievalParams());
  utils::ScopeDeleter1<IVFPQRetrievalParameters> del_params;
  if (retrieval_params == nullptr) {
    retrieval_params = new IVFPQRetrievalParameters();
    del_params.set(retrieval_params);
  }

  faiss::MetricType metric_type;
  if (retrieval_params->GetDistanceComputeType() ==
      DistanceComputeType::INNER_PRODUCT) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }

  size_t raw_d = mem_raw_vec->MetaInfo()->Dimension();

  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  bool parallel_mode = retrieval_params->ParallelOnQueries() ? 0 : 1;

  // don't start parallel section if single query
  bool do_parallel = parallel_mode == 0 ? n > 1 : nprobe > 1;

  size_t ndis = 0;
#pragma omp parallel if (do_parallel) reduction(+ : ndis)
  {
    GammaInvertedListScanner *scanner =
        GetGammaIVFFlatScanner(raw_d, metric_type);
    utils::ScopeDeleter1<GammaInvertedListScanner> del(scanner);
    scanner->set_search_context(retrieval_context);

    /****************************************************
     * Actual loops, depending on parallel_mode
     ****************************************************/

    if (parallel_mode == 0) {  // parallelize over queries

#pragma omp for
      for (int i = 0; i < n; i++) {
        // loop over queries
        scanner->set_query(x + i * d);
        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        init_result(metric_type, k, simi, idxi);

        size_t nscan = 0;

        // loop over probes
        for (int ik = 0; ik < nprobe; ik++) {
          nscan += scan_one_list(scanner, keys[i * nprobe + ik],
                                 coarse_dis[i * nprobe + ik], simi, idxi, k,
                                 this->nlist, this->invlists, store_pairs,
                                 retrieval_params->IvfFlat(), mem_raw_vec);

          if (max_codes && nscan >= max_codes) {
            break;
          }
        }

        ndis += nscan;
        reorder_result(metric_type, k, simi, idxi);
      }       // parallel for
    } else {  // parallelize over inverted lists
      std::vector<idx_t> local_idx(k);
      std::vector<float> local_dis(k);

      for (int i = 0; i < n; i++) {
        scanner->set_query(x + i * d);
        init_result(metric_type, k, local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
        for (int ik = 0; ik < nprobe; ik++) {
          ndis += scan_one_list(scanner, keys[i * nprobe + ik],
                                coarse_dis[i * nprobe + ik], local_dis.data(),
                                local_idx.data(), k, this->nlist,
                                this->invlists, store_pairs,
                                retrieval_params->IvfFlat(), mem_raw_vec);

          // can't do the test on max_codes
        }
        // merge thread-local results

        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;
#pragma omp single
        init_result(metric_type, k, simi, idxi);

#pragma omp barrier
#pragma omp critical
        {
          if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis.data(),
                                        local_idx.data(), k);
          } else {
            faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis.data(),
                                        local_idx.data(), k);
          }
        }
#pragma omp barrier
#pragma omp single
        reorder_result(metric_type, k, simi, idxi);
      }
    }
  }  // parallel section
#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "ivf flat compute ";
  compute_msg += std::to_string(n);
  retrieval_context->GetPerfTool().Perf(compute_msg);
#endif
}

void GammaIVFPQIndex::search_preassigned(
    RetrievalContext *retrieval_context, int n, const float *x, const float *applied_x, int k,
    const idx_t *keys, const float *coarse_dis, float *distances, idx_t *labels,
    int nprobe, bool store_pairs, const faiss::IVFSearchParameters *params) {
  int raw_d = vector_->MetaInfo()->Dimension();
  // for opq, rerank need raw vector
  float *vec_q = nullptr;
  utils::ScopeDeleter1<float> del_vec_q;
  if (d > raw_d) {
    float *vec = new float[n * d];

    ConvertVectorDim(n, raw_d, d, x, vec);

    vec_q = vec;
    del_vec_q.set(vec_q);
  } else {
    vec_q = const_cast<float *>(x);
  }
  
  float *vec_applied_q = nullptr;
  utils::ScopeDeleter1<float> del_applied_q;
  if (d > raw_d) {
    float *applied_vec = new float[n * d];

    ConvertVectorDim(n, raw_d, d, applied_x, applied_vec);

    vec_applied_q = applied_vec;
    del_applied_q.set(vec_applied_q);
  } else {
    vec_applied_q = const_cast<float *>(applied_x);
  }

  GammaSearchCondition *context =
      dynamic_cast<GammaSearchCondition *>(retrieval_context);
  IVFPQRetrievalParameters *retrieval_params =
      dynamic_cast<IVFPQRetrievalParameters *>(
          retrieval_context->RetrievalParams());
  utils::ScopeDeleter1<IVFPQRetrievalParameters> del_params;
  if (retrieval_params == nullptr) {
    retrieval_params = new IVFPQRetrievalParameters();
    del_params.set(retrieval_params);
  }

  faiss::MetricType metric_type;
  if (retrieval_params->GetDistanceComputeType() ==
      DistanceComputeType::INNER_PRODUCT) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }
  long max_codes = params ? params->max_codes : this->max_codes;

  if (k <= 0) {
    LOG(WARNING) << "topK is should greater then 0, topK = " << k;
    return;
  }
  size_t ndis = 0;

  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  int recall_num = retrieval_params->RecallNum();
  if (recall_num < k) {
    recall_num = k;
  }

  float *recall_distances = new float[n * recall_num];
  idx_t *recall_labels = new idx_t[n * recall_num];
  utils::ScopeDeleter<float> del1(recall_distances);
  utils::ScopeDeleter<idx_t> del2(recall_labels);

#ifdef PERFORMANCE_TESTING
  retrieval_context->GetPerfTool().Perf("search prepare");
#endif

  bool parallel_mode = retrieval_params->ParallelOnQueries() ? 0 : 1;

  // don't start parallel section if single query
  bool do_parallel = parallel_mode == 0 ? n > 1 : nprobe > 1;

#pragma omp parallel if (do_parallel) reduction(+ : ndis)
  {
    GammaInvertedListScanner *scanner =
        GetGammaInvertedListScanner(store_pairs, metric_type);
    utils::ScopeDeleter1<GammaInvertedListScanner> del(scanner);
    scanner->set_search_context(retrieval_context);

    if (parallel_mode == 0) {  // parallelize over queries
#pragma omp for
      for (int i = 0; i < n; i++) {
        // loop over queries
        const float *xi = vec_applied_q + i * d;
        scanner->set_query(xi);
        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        float *recall_simi = recall_distances + i * recall_num;
        idx_t *recall_idxi = recall_labels + i * recall_num;

        init_result(metric_type, k, simi, idxi);
        init_result(metric_type, recall_num, recall_simi, recall_idxi);

        long nscan = 0;

        // loop over probes
        for (int ik = 0; ik < nprobe; ik++) {
          nscan += scan_one_list(
              scanner, keys[i * nprobe + ik], coarse_dis[i * nprobe + ik],
              recall_simi, recall_idxi, recall_num, this->nlist, this->invlists,
              store_pairs, retrieval_params->IvfFlat());

          if (max_codes && nscan >= max_codes) break;
        }

        ndis += nscan;
        compute_dis(k, vec_q + i * d, simi, idxi, recall_simi, recall_idxi, recall_num,
                    context->has_rank, metric_type, vector_, retrieval_context);
      }       // parallel for
    } else {  // parallelize over inverted lists
      std::vector<idx_t> local_idx(recall_num);
      std::vector<float> local_dis(recall_num);

      for (int i = 0; i < n; i++) {
        const float *xi = vec_applied_q + i * d;
        scanner->set_query(xi);

        init_result(metric_type, recall_num, local_dis.data(),
                    local_idx.data());

#pragma omp for schedule(dynamic)
        for (int ik = 0; ik < nprobe; ik++) {
          ndis += scan_one_list(
              scanner, keys[i * nprobe + ik], coarse_dis[i * nprobe + ik],
              local_dis.data(), local_idx.data(), recall_num, this->nlist,
              this->invlists, store_pairs, retrieval_params->IvfFlat());

          // can't do the test on max_codes
        }

        // merge thread-local results

        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        float *recall_simi = recall_distances + i * recall_num;
        idx_t *recall_idxi = recall_labels + i * recall_num;

#pragma omp single
        {
          init_result(metric_type, k, simi, idxi);
          init_result(metric_type, recall_num, recall_simi, recall_idxi);
        }

#pragma omp barrier
#pragma omp critical
        {
          if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            faiss::heap_addn<HeapForIP>(recall_num, recall_simi, recall_idxi,
                                        local_dis.data(), local_idx.data(),
                                        recall_num);
          } else {
            faiss::heap_addn<HeapForL2>(recall_num, recall_simi, recall_idxi,
                                        local_dis.data(), local_idx.data(),
                                        recall_num);
          }
        }
#pragma omp barrier
#pragma omp single
        {
#ifdef PERFORMANCE_TESTING
          retrieval_context->GetPerfTool().Perf("coarse");
#endif
          compute_dis(k, vec_q + i * d, simi, idxi, recall_simi, recall_idxi, recall_num,
                      context->has_rank, metric_type, vector_,
                      retrieval_context);

#ifdef PERFORMANCE_TESTING
          retrieval_context->GetPerfTool().Perf("reorder");
#endif
        }
      }
    }
  }  // parallel

#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "compute ";
  compute_msg += std::to_string(n);
  retrieval_context->GetPerfTool().Perf(compute_msg);
#endif
}  // namespace tig_gamma

void GammaIVFPQIndex::copy_subset_to(faiss::IndexIVF &other, int subset_type,
                                     idx_t a1, idx_t a2) const {
  using ScopedIds = faiss::InvertedLists::ScopedIds;
  using ScopedCodes = faiss::InvertedLists::ScopedCodes;
  FAISS_THROW_IF_NOT(nlist == other.nlist);
  FAISS_THROW_IF_NOT(code_size == other.code_size);
  // FAISS_THROW_IF_NOT(other.direct_map.no());
  FAISS_THROW_IF_NOT_FMT(
      subset_type == 0 || subset_type == 1 || subset_type == 2,
      "subset type %d not implemented", subset_type);

  int accu_n = 0;

  faiss::InvertedLists *oivf = other.invlists;

  for (size_t list_no = 0; list_no < nlist; list_no++) {
    size_t n = invlists->list_size(list_no);
    ScopedIds ids_in(invlists, list_no);

    if (subset_type == 0) {
      for (size_t i = 0; i < n; i++) {
        idx_t id = ids_in[i];
        if (a1 <= id && id < a2) {
          oivf->add_entry(list_no, invlists->get_single_id(list_no, i),
                          ScopedCodes(invlists, list_no, i).get());
          other.ntotal++;
        }
      }
    } else if (subset_type == 1) {
      for (size_t i = 0; i < n; i++) {
        idx_t id = ids_in[i];
        if (id % a1 == a2) {
          oivf->add_entry(list_no, invlists->get_single_id(list_no, i),
                          ScopedCodes(invlists, list_no, i).get());
          other.ntotal++;
        }
      }
    }
    accu_n += n;
  }
  // FAISS_ASSERT(accu_n == indexed_vec_count_);
}

string IVFPQToString(const faiss::IndexIVFPQ *ivpq, const faiss::VectorTransform *vt) {
  std::stringstream ss;
  ss << "d=" << ivpq->d << ", ntotal=" << ivpq->ntotal
     << ", is_trained=" << ivpq->is_trained
     << ", metric_type=" << ivpq->metric_type << ", nlist=" << ivpq->nlist
     << ", nprobe=" << ivpq->nprobe << ", by_residual=" << ivpq->by_residual
     << ", code_size=" << ivpq->code_size << ", pq: d=" << ivpq->pq.d
     << ", M=" << ivpq->pq.M << ", nbits=" << ivpq->pq.nbits;

  faiss::IndexHNSWFlat *hnsw_flat = dynamic_cast<faiss::IndexHNSWFlat *>(ivpq->quantizer);
  if (hnsw_flat) {
    ss << ", hnsw: efSearch=" << hnsw_flat->hnsw.efSearch
       << ", efConstruction=" << hnsw_flat->hnsw.efConstruction
       << ", search_bounded_queue=" << hnsw_flat->hnsw.search_bounded_queue;
  }

  const faiss::OPQMatrix *opq = dynamic_cast<const faiss::OPQMatrix *>(vt);
  if (opq) {
    ss << ", opq: d_in=" << opq->d_in << ", d_out=" << opq->d_out << ", M=" << opq->M;
  }
  return ss.str();
}

int GammaIVFPQIndex::Dump(const std::string &dir) {
  if (!this->is_trained) {
    LOG(INFO) << "gamma index is not trained, skip dumping";
    return 0;
  }
  std::string index_name = vector_->MetaInfo()->AbsoluteName();
  string index_dir = dir + "/" + index_name;
  if (utils::make_dir(index_dir.c_str())) {
    LOG(ERROR) << "mkdir error, index dir=" << index_dir;
    return IO_ERR;
  }

  string index_file = index_dir + "/ivfpq.index";
  faiss::IOWriter *f = new FileIOWriter(index_file.c_str());
  utils::ScopeDeleter1<FileIOWriter> del((FileIOWriter *)f);
  const IndexIVFPQ *ivpq = static_cast<const IndexIVFPQ *>(this);
  uint32_t h = faiss::fourcc("IwPQ");
  WRITE1(h);
  tig_gamma::write_ivf_header(ivpq, f);
  WRITE1(ivpq->by_residual);
  WRITE1(ivpq->code_size);
  tig_gamma::write_product_quantizer(&ivpq->pq, f);

  if (opq_ != nullptr)
    write_opq(opq_, f);

  int indexed_count = indexed_vec_count_;
  if (WriteInvertedLists(f, rt_invert_index_ptr_)) {
    LOG(ERROR) << "write invert list error, index name=" << index_name;
    return INTERNAL_ERR;
  }
  WRITE1(indexed_count);

  LOG(INFO) << "dump:" << IVFPQToString(ivpq, opq_) << ", indexed count=" << indexed_count;
  return 0;
}

int GammaIVFPQIndex::Load(const std::string &index_dir) {
  std::string index_name = vector_->MetaInfo()->AbsoluteName();
  string index_file = index_dir + "/" + index_name + "/ivfpq.index";
  if (!utils::file_exist(index_file)) {
    LOG(INFO) << index_file << " isn't existed, skip loading";
    return 0;  // it should train again after load
  }

  faiss::IOReader *f = new FileIOReader(index_file.c_str());
  utils::ScopeDeleter1<FileIOReader> del((FileIOReader *)f);
  uint32_t h;
  READ1(h);
  assert(h == faiss::fourcc("IwPQ"));
  IndexIVFPQ *ivpq = static_cast<IndexIVFPQ *>(this);
  tig_gamma::read_ivf_header(ivpq, f, nullptr);  // not legacy
  READ1(ivpq->by_residual);
  READ1(ivpq->code_size);
  tig_gamma::read_product_quantizer(&ivpq->pq, f);

  faiss::IndexHNSWFlat *hnsw_flat = dynamic_cast<faiss::IndexHNSWFlat *>(ivpq->quantizer);
  if(hnsw_flat) {
    hnsw_flat->hnsw.search_bounded_queue = false;
    quantizer_type_ = 1;
  }
  if(opq_) {
    read_opq(opq_, f);
  }

  int ret = ReadInvertedLists(f, rt_invert_index_ptr_);
  if (ret == FORMAT_ERR) {
    indexed_vec_count_ = 0;
    LOG(INFO) << "unsupported inverted list format, it need rebuilding!";
  } else if (ret == 0) {
    READ1(indexed_vec_count_);
    if (indexed_vec_count_ < 0 ||
        indexed_vec_count_ > vector_->MetaInfo()->size_) {
      LOG(ERROR) << "invalid indexed count=" << indexed_vec_count_;
      return INTERNAL_ERR;
    }
    // precomputed table not stored. It is cheaper to recompute it
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual) ivpq->precompute_table();
    LOG(INFO) << "load: " << IVFPQToString(ivpq, opq_)
              << ", indexed vector count=" << indexed_vec_count_;
  } else {
    LOG(ERROR) << "read invert list error, index name=" << index_name;
    return INTERNAL_ERR;
  }
  if (ivpq->metric_type == faiss::METRIC_INNER_PRODUCT) {
    metric_type_ = DistanceComputeType::INNER_PRODUCT;
  } else {
    metric_type_ = DistanceComputeType::L2;
  }
  assert(this->is_trained);
  return indexed_vec_count_;
}

}  // namespace tig_gamma
