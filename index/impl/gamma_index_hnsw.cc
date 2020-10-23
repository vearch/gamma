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

#include "gamma_index_hnsw.h"

#include <immintrin.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

#include "error_code.h"
#include "memory_raw_vector.h"
#include "utils.h"

namespace tig_gamma {

using idx_t = faiss::Index::idx_t;
using MinimaxHeap = faiss::HNSW::MinimaxHeap;
using storage_idx_t = faiss::HNSW::storage_idx_t;
using NodeDistCloser = faiss::HNSW::NodeDistCloser;
using NodeDistFarther = faiss::HNSW::NodeDistFarther;
using RandomGenerator = faiss::RandomGenerator;
using DistanceComputer = faiss::DistanceComputer;
using ReconstructFromNeighbors = faiss::ReconstructFromNeighbors;

struct HNSWModelParams {
  int nlinks;          // link number for hnsw graph
  int efConstruction;  // construction parameter for building hnsw graph
  DistanceComputeType metric_type;
 
  HNSWModelParams() {
    nlinks = 32;
    efConstruction = 40;
    metric_type = DistanceComputeType::L2;
  }

  bool Validate() {
    if (nlinks < 0 || efConstruction < 0) return false;
    return true;
  }

  int Parse(const char *str) {
    utils::JsonParser jp;
    if (jp.Parse(str)) {
      LOG(ERROR) << "parse HNSW retrieval parameters error: " << str;
      return -1;
    }

    int nlinks;
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
    } else {
      this->metric_type = DistanceComputeType::L2;
    }

    return 0;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "nlinks =" << nlinks << ", ";
    ss << "efConstruction =" << efConstruction << ", ";
    ss << "metric_type =" << (int)metric_type;
    return ss.str();
  }
};

REGISTER_MODEL(HNSW, GammaHNSWIndex)

GammaHNSWIndex::GammaHNSWIndex()
    : GammaFLATIndex(), faiss::IndexHNSW(0, 32), gamma_hnsw_(32) {
  indexed_vec_count_ = 0;
  has_update_ = false;
}

GammaHNSWIndex::~GammaHNSWIndex() {
  for (size_t i = 0; i < locks_.size(); i++) {
    omp_destroy_lock(&locks_[i]);
  }
  int ret = pthread_rwlock_destroy(&mutex_);
  if (0 != ret) {
    LOG(ERROR) << "destory read write lock error, ret=" << ret;
  }
}

int GammaHNSWIndex::Init(const std::string &model_parameters) {
  auto raw_vec_type = dynamic_cast<MemoryRawVector *>(vector_);
  if (raw_vec_type == nullptr) {
    LOG(ERROR) << "HNSW can only work in memory only mode";
    return -1;
  }

  HNSWModelParams hnsw_param;
  if (model_parameters != "" && hnsw_param.Parse(model_parameters.c_str())) {
    return -2;
  }
  LOG(INFO) << hnsw_param.ToString();

  d = vector_->MetaInfo()->Dimension();
  // reset hnsw
  gamma_hnsw_.cum_nneighbor_per_level.clear();
  gamma_hnsw_.assign_probas.clear();
  gamma_hnsw_.set_default_probas(hnsw_param.nlinks,
                                 1.0 / log(hnsw_param.nlinks));
  gamma_hnsw_.efConstruction = hnsw_param.efConstruction;
 
  if (hnsw_param.metric_type ==
      DistanceComputeType::INNER_PRODUCT) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }

  int ret = pthread_rwlock_init(&mutex_, NULL);
  if (ret != 0) {
    LOG(ERROR) << "init read-write lock error, ret=" << ret;
    return -3;
  }
  return 0;
}

RetrievalParameters *GammaHNSWIndex::Parse(const std::string &parameters) {
  enum DistanceComputeType type;
  if(this->metric_type == faiss::METRIC_L2) {
    type = DistanceComputeType::L2;
  } else {
    type = DistanceComputeType::INNER_PRODUCT;
  }

  if (parameters == "") {
    return new HnswRetrievalParameters(type);
  }

  utils::JsonParser jp;
  if (jp.Parse(parameters.c_str())) {
    LOG(ERROR) << "parse retrieval parameters error: " << parameters;
    return nullptr;
  }

  std::string metric_type;
  if (!jp.GetString("metric_type", metric_type)) {
    if (strcasecmp("L2", metric_type.c_str()) &&
        strcasecmp("InnerProduct", metric_type.c_str())) {
      LOG(ERROR) << "invalid metric_type = " << metric_type
                 << ", so use default value.";
    }
    if (!strcasecmp("L2", metric_type.c_str()))
      type = DistanceComputeType::L2;
    else
      type = DistanceComputeType::INNER_PRODUCT;
  }

  int efSearch = 0;
  jp.GetInt("efSearch", efSearch);

  RetrievalParameters *retrieval_params =
      new HnswRetrievalParameters(efSearch > 0 ? efSearch : 64, type);
  return retrieval_params;
}

int GammaHNSWIndex::Indexing() {
  return 0;
}

bool GammaHNSWIndex::Add(int n, const uint8_t *vec) {
  int n0 = indexed_vec_count_;
  const float *x = reinterpret_cast<const float *>(vec);

  std::unique_lock<std::mutex> templock(dump_mutex_);
  AddVertices(n0, n, x, verbose, gamma_hnsw_.levels.size() == (size_t)(n0 + n));
  ntotal = n0 + n;
  indexed_vec_count_ += n;

  return true;
}

namespace {

struct FlatL2Dis : DistanceComputer {
  size_t d;
  const RawVector *raw_vec;
  const float *q;
  size_t ndis;

  float operator()(idx_t i) override {
    ndis++;
    ScopeVector svec;
    raw_vec->GetVector(i, svec);
    return faiss::fvec_L2sqr(q, (const float *)svec.Get(), d);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    ndis++;
    ScopeVector svecx, svecy;
    raw_vec->GetVector(i, svecx);
    raw_vec->GetVector(j, svecy);
    return faiss::fvec_L2sqr((const float *)svecx.Get(),
                             (const float *)svecy.Get(), d);
  }

  explicit FlatL2Dis(size_t d, const MemoryRawVector *raw_vec,
                     const float *q = nullptr)
      : d(d), raw_vec(raw_vec), q(q), ndis(0) {}

  void set_query(const float *x) override { q = x; }
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct FlatIPDis : DistanceComputer {
  size_t d;
  const RawVector *raw_vec;
  idx_t nb;
  const float *q;
  size_t ndis;

  float operator()(idx_t i) override {
    ndis++;
    ScopeVector svec;
    raw_vec->GetVector(i, svec);
    return -faiss::fvec_inner_product(q, (const float *)svec.Get(), d);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    ScopeVector svecx, svecy;
    raw_vec->GetVector(i, svecx);
    raw_vec->GetVector(j, svecy);
    return -faiss::fvec_inner_product((const float *)svecx.Get(),
                                      (const float *)svecy.Get(), d);
  }

  explicit FlatIPDis(size_t d, const MemoryRawVector *raw_vec,
                     const float *q = nullptr)
      : d(d), raw_vec(raw_vec), q(q), ndis(0) {}

  void set_query(const float *x) override { q = x; }
};

};  // namespace

DistanceComputer *GammaHNSWIndex::GetDistanceComputer(
    faiss::MetricType metric_type) const {
  if (metric_type == faiss::METRIC_L2) {
    return new FlatL2Dis(d, dynamic_cast<const MemoryRawVector *>(vector_));
  } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    return new FlatIPDis(d, dynamic_cast<const MemoryRawVector *>(vector_));
  }

  return nullptr;
}

int GammaHNSWIndex::AddVertices(size_t n0, size_t n, const float *x,
                                bool verbose, bool preset_levels) {
#ifdef PERFORMANCE_TESTING
  double t0 = utils::getmillisecs();
#endif  // PERFORMANCE_TESTING
  if (n == 0) {
    return 0;
  }
  pthread_rwlock_wrlock(&mutex_);
  gamma_hnsw_.prepare_level_tab(n, preset_levels);
  pthread_rwlock_unlock(&mutex_);

  // add vectors from highest to lowest level
  std::vector<int> hist;
  std::vector<int> order(n);

  // add lock for each node
  std::vector<omp_lock_t> tmp_locks(n);
  locks_.insert(locks_.end(), tmp_locks.data(), tmp_locks.data() + n);

  for (size_t i = 0; i < n; i++) {
    omp_init_lock(&locks_[i + n0]);
  }

  {  // make buckets with vectors of the same level

    // build histogram
    for (size_t i = 0; i < n; i++) {
      storage_idx_t pt_id = i + n0;
      size_t pt_level = gamma_hnsw_.levels[pt_id] - 1;
      while (pt_level >= hist.size()) hist.push_back(0);
      hist[pt_level]++;
    }

    // accumulate
    std::vector<int> offsets(hist.size() + 1, 0);
    for (size_t i = 0; i < hist.size() - 1; i++) {
      offsets[i + 1] = offsets[i] + hist[i];
    }

    // bucket sort
    for (size_t i = 0; i < n; i++) {
      storage_idx_t pt_id = i + n0;
      int pt_level = gamma_hnsw_.levels[pt_id] - 1;
      order[offsets[pt_level]++] = pt_id;
    }
  }

  {  // perform add
    RandomGenerator rng2(789);

    int i1 = n;

    for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
      int i0 = i1 - hist[pt_level];

      if (verbose) {
        LOG(INFO) << "Adding " << i1 - i0 << " elements at level " << pt_level;
      }

      // random permutation to get rid of dataset order bias
      for (int j = i0; j < i1; j++)
        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

      int num_threads =
          (i1 - i0) < omp_get_max_threads() ? (i1 - i0) : omp_get_max_threads();

#pragma omp parallel if (i1 > i0 + 10) num_threads(num_threads)
      {
        DistanceComputer *dis = GetDistanceComputer(metric_type);
        faiss::ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for schedule(dynamic)
        for (int i = i0; i < i1; i++) {
          storage_idx_t pt_id = order[i];
          dis->set_query(x + (pt_id - n0) * d);
          gamma_hnsw_.AddWithLocks(*dis, pt_level, pt_id, locks_);
        }
      }
      i1 = i0;
    }
    FAISS_ASSERT(i1 == 0);
  }

#ifdef PERFORMANCE_TESTING
  add_count_ += n;
  if (add_count_ >= 10000) {
    LOG(INFO) << "adding elements on top of " << n0 << ", average add time "
              << (utils::getmillisecs() - t0) / n << " ms";
    add_count_ = 0;
  }
#endif  // PERFORMANCE_TESTING

  return 0;
}

int GammaHNSWIndex::Search(RetrievalContext *retrieval_context, int n,
                           const uint8_t *x, int k, float *distances,
                           int64_t *labels) {
  int num_threads = n < omp_get_max_threads() ? n : omp_get_max_threads();

  const float *xq = reinterpret_cast<const float *>(x);
  if (xq == nullptr) {
    LOG(ERROR) << "search feature is null";
    return -1;
  }

  idx_t *idxs = reinterpret_cast<idx_t *>(labels);
  if (idxs == nullptr) {
    LOG(ERROR) << "search result'ids is null";
    return -2;
  }

  HnswRetrievalParameters *retrieval_params =
      dynamic_cast<HnswRetrievalParameters *>(
          retrieval_context->RetrievalParams());
  utils::ScopeDeleter1<HnswRetrievalParameters> del_params;
  if (retrieval_params == nullptr) {
    retrieval_params = new HnswRetrievalParameters();
    del_params.set(retrieval_params);
  }

  faiss::MetricType metric_type;
  if (retrieval_params->GetDistanceComputeType() ==
      DistanceComputeType::INNER_PRODUCT) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }

#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
  for (int i = 0; i < n; i++) {
    DistanceComputer *dis = GetDistanceComputer(metric_type);
    faiss::ScopeDeleter1<DistanceComputer> del(dis);
    idx_t *idxi = idxs + i * k;
    float *simi = distances + i * k;
    dis->set_query(xq + i * d);

    faiss::maxheap_heapify(k, simi, idxi);

    pthread_rwlock_rdlock(&mutex_);
    gamma_hnsw_.Search(*dis, k, idxi, simi, retrieval_params->EfSearch(),
                       retrieval_context);
    pthread_rwlock_unlock(&mutex_);

    faiss::maxheap_reorder(k, simi, idxi);

    if (reconstruct_from_neighbors &&
        reconstruct_from_neighbors->k_reorder != 0) {
      int k_reorder = reconstruct_from_neighbors->k_reorder;
      if (k_reorder == -1 || k_reorder > k) k_reorder = k;

      reconstruct_from_neighbors->compute_distances(k_reorder, idxi, xq + i * d,
                                                    simi);

      // sort top k_reorder
      faiss::maxheap_heapify(k_reorder, simi, idxi, simi, idxi, k_reorder);
      faiss::maxheap_reorder(k_reorder, simi, idxi);
    }
  }

  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    // we need to revert the negated distances
    for (int i = 0; i < k * n; i++) {
      distances[i] = -distances[i];
    }
  }

#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "hnsw compute ";
  compute_msg += std::to_string(n);
  retrieval_context->GetPerfTool().Perf(compute_msg);
#endif  // PERFORMANCE_TESTING

  return 0;
}

long GammaHNSWIndex::GetTotalMemBytes() {
  size_t total_mem_bytes = 0;
  total_mem_bytes += locks_.size() * sizeof(omp_lock_t);

  total_mem_bytes += gamma_hnsw_.assign_probas.size() * sizeof(double);
  total_mem_bytes += gamma_hnsw_.cum_nneighbor_per_level.size() * sizeof(int);
  total_mem_bytes += gamma_hnsw_.levels.size() * sizeof(int);
  total_mem_bytes += gamma_hnsw_.offsets.size() * sizeof(size_t);
  total_mem_bytes += gamma_hnsw_.neighbors.size() * sizeof(storage_idx_t);

  total_mem_bytes += sizeof(gamma_hnsw_.entry_point);
  total_mem_bytes += sizeof(gamma_hnsw_.max_level);
  total_mem_bytes += sizeof(gamma_hnsw_.nlinks);
  total_mem_bytes += sizeof(gamma_hnsw_.efConstruction);
  total_mem_bytes += sizeof(gamma_hnsw_.efSearch);
  total_mem_bytes += sizeof(gamma_hnsw_.upper_beam);

  return total_mem_bytes;
}

// TODO
int GammaHNSWIndex::Update(const std::vector<int64_t> &ids,
                           const std::vector<const uint8_t *> &vecs) {
  std::unique_lock<std::mutex> templock(dump_mutex_);
  has_update_ = true;
  return 0;
}

int GammaHNSWIndex::Delete(const std::vector<int64_t> &ids) { return 0; }

std::string HNSWToString(const GammaHNSWIndex *idxhnsw) {
  std::stringstream ss;
  ss << "d=" << idxhnsw->d << ", ntotal=" << idxhnsw->ntotal
     << ", is_trained=" << idxhnsw->is_trained
     << ", metric_type=" << idxhnsw->metric_type
     << ", nlinks=" << idxhnsw->gamma_hnsw_.nlinks
     << ", efSearch=" << idxhnsw->gamma_hnsw_.efSearch
     << ", efConstruction=" << idxhnsw->gamma_hnsw_.efConstruction;
  return ss.str();
}

int GammaHNSWIndex::Dump(const std::string &dir) {
  if (!this->is_trained) {
    LOG(INFO) << "gamma hnsw index is not trained, skip dumping";
    return 0;
  }
  std::string index_name = vector_->MetaInfo()->AbsoluteName();
  string index_dir = dir + "/" + index_name;
  if (utils::make_dir(index_dir.c_str())) {
    LOG(ERROR) << "mkdir error, index dir=" << index_dir;
    return IO_ERR;
  }

  string index_file = index_dir + "/hnsw.index";
  faiss::IOWriter *f = new FileIOWriter(index_file.c_str());
  utils::ScopeDeleter1<FileIOWriter> del((FileIOWriter *)f);

  std::unique_lock<std::mutex> templock(dump_mutex_);
  int indexed_count = indexed_vec_count_;
  const GammaHNSWIndex * idxhnsw = dynamic_cast<const GammaHNSWIndex *> (this);
  uint32_t h = faiss::fourcc("IHNf");
  WRITE1 (h);
  write_index_header (idxhnsw, f);
  write_hnsw (&idxhnsw->gamma_hnsw_, f);
  WRITE1(indexed_count);
  WRITE1(has_update_);
  LOG(INFO) << "dump:" << HNSWToString(idxhnsw) << ", indexed count=" << indexed_count;
  return 0;
}

int GammaHNSWIndex::Load(const std::string &index_dir) {
  std::string index_name = vector_->MetaInfo()->AbsoluteName();
  string index_file = index_dir + "/" + index_name + "/hnsw.index";
  if (!utils::file_exist(index_file)) {
    LOG(INFO) << index_file << " isn't existed, skip loading";
    return 0;  // it should train again after load
  }

  faiss::IOReader *f = new FileIOReader(index_file.c_str());
  utils::ScopeDeleter1<FileIOReader> del((FileIOReader *)f);
  uint32_t h;
  READ1(h);
  assert(h == faiss::fourcc("IHNf"));
  GammaHNSWIndex *idxhnsw = static_cast<GammaHNSWIndex *>(this);
  read_index_header (idxhnsw, f);
  read_hnsw (&idxhnsw->gamma_hnsw_, f);

  READ1(indexed_vec_count_);
  if (indexed_vec_count_ < 0 ||
      (size_t) indexed_vec_count_ > vector_->MetaInfo()->size_) {
    LOG(ERROR) << "invalid indexed count=" << indexed_vec_count_;
    return INTERNAL_ERR;
  }
  assert(this->is_trained);

  if (indexed_vec_count_ != idxhnsw->ntotal) {
     LOG(ERROR) << "indexed count=" << indexed_vec_count_
                << "should be equal to ntotal=" << idxhnsw->ntotal;
     return INTERNAL_ERR;
  }
  READ1(has_update_);
  if (has_update_ == true) {
    indexed_vec_count_ = 0;
    idxhnsw->gamma_hnsw_.reset();
    idxhnsw->ntotal = 0;
    LOG(INFO) << "hnsw has been updated, so reset it to rebuild the graph.";
  }
  LOG(INFO) << "load: " << HNSWToString(idxhnsw)
            << ", indexed vector count=" << indexed_vec_count_;

  RawVector *raw_vec = dynamic_cast<RawVector *>(vector_);
  raw_vec->SetIndexedVectorNum(indexed_vec_count_);

  omp_lock_t tmp_lock;
  locks_.insert(locks_.end(), indexed_vec_count_, tmp_lock);
  for(int i = 0; i < indexed_vec_count_; i++) {
    omp_init_lock(&locks_[i]);
  }
  return indexed_vec_count_;
}

GammaHNSWFlatIndex::GammaHNSWFlatIndex() : GammaHNSWIndex() {
  is_trained = true;
}

}  // namespace tig_gamma
