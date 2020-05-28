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
#include <cstdlib>
#include <unistd.h>
#include <immintrin.h>

namespace tig_gamma {

using idx_t = faiss::Index::idx_t;
using MinimaxHeap = faiss::HNSW::MinimaxHeap;
using storage_idx_t = faiss::HNSW::storage_idx_t;
using NodeDistCloser = faiss::HNSW::NodeDistCloser;
using NodeDistFarther = faiss::HNSW::NodeDistFarther;
using RandomGenerator = faiss::RandomGenerator;
using DistanceComputer = faiss::DistanceComputer;
using ReconstructFromNeighbors = faiss::ReconstructFromNeighbors;

GammaHNSWIndex::GammaHNSWIndex(faiss::Index *quantizer, size_t d, 
                              DistanceMetricType metric_type,
                              int M, int efSearch, int efConstruction,
                              const char *docids_bitmap, RawVector<float> *raw_vec)
    : GammaFLATIndex(d, docids_bitmap, raw_vec), 
      faiss::IndexHNSW(quantizer, M), gamma_hnsw_(M) {

  assert(raw_vec != nullptr);

  this->d = d;
  raw_vec_head_ = nullptr;
  indexed_vec_count_ = 0;

  gamma_hnsw_.efSearch = efSearch;
  gamma_hnsw_.efConstruction = efConstruction;

  if (metric_type == InnerProduct) {
    this->metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    this->metric_type = faiss::METRIC_L2;
  }
  int ret = pthread_rwlock_init(&mutex_, NULL);
  if (ret != 0) {
    LOG(ERROR) << "init read-write lock error, ret=" << ret;
  }
}

GammaHNSWIndex::~GammaHNSWIndex() {
  for(int i = 0; i < indexed_vec_count_; i++) {
    omp_destroy_lock(&locks_[i]);
  }
  int ret = pthread_rwlock_destroy(&mutex_);
  if (0 != ret) {
    LOG(ERROR) << "destory read write lock error, ret=" << ret;
  }
}

int GammaHNSWIndex::Indexing() {
  // get raw vec head for DistanceComputer
  ScopeVector<float> vector_head;
  raw_vec_->GetVectorHeader(0, 0, vector_head);
  raw_vec_head_ = const_cast<float *>(vector_head.Get());
  return 0;
}

int GammaHNSWIndex::AddRTVecsToIndex() {
  int ret = 0;
  int total_stored_vecs = raw_vec_->GetVectorNum();
  if (indexed_vec_count_ > total_stored_vecs) {
    LOG(ERROR) << "internal error : indexed_vec_count=" << indexed_vec_count_
               << " should not greater than total_stored_vecs="
               << total_stored_vecs;
    ret = -1;
  } else if (indexed_vec_count_ == total_stored_vecs) {
    ;
#ifdef DEBUG
    LOG(INFO) << "no extra vectors existed for indexing";
#endif
  } else {

    int MAX_NUM_PER_INDEX = 1000;
    int index_count =
        (total_stored_vecs - indexed_vec_count_) / MAX_NUM_PER_INDEX + 1;

    for (int i = 0; i < index_count; i++) {
      int start_docid = indexed_vec_count_;
      int count_per_index =
          (i == (index_count - 1) ? total_stored_vecs - start_docid
                                  : MAX_NUM_PER_INDEX);
      ScopeVector<float> vector_head;
      raw_vec_->GetVectorHeader(indexed_vec_count_,
                                indexed_vec_count_ + count_per_index,
                                vector_head);

      float *add_vec = const_cast<float *>(vector_head.Get());
      if (!Add(count_per_index, add_vec)) {
        LOG(ERROR) << "add index from docid " << start_docid << " error!";
        ret = -2;
      }
    }
  }
  return ret;      
}

bool GammaHNSWIndex::Add(int n, const float *vec) {
  int n0 = indexed_vec_count_;
  ntotal = n0 + n;
  AddVertices(n0, n, vec, verbose,
              gamma_hnsw_.levels.size() == (size_t)ntotal);

  indexed_vec_count_ += n;

  return true;
}

namespace {

struct FlatL2Dis : DistanceComputer {
  size_t d;
  idx_t nb;
  const float *xb;
  const float *q;
  size_t ndis;

  float operator () (idx_t i) override {
    ndis++;
    return faiss::fvec_L2sqr(q, xb + i * d, d);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    ndis++;
    return faiss::fvec_L2sqr(xb + j * d, xb + i * d, d);
  }

  explicit FlatL2Dis(size_t d, idx_t nb, 
                    const float *xb = nullptr,
                    const float *q = nullptr)
      : d(d),
        nb(nb),
        xb(xb),
        q(q),
        ndis(0) {}

  void set_query(const float *x) override {
      q = x;
  }
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct FlatIPDis : DistanceComputer {
  size_t d;
  idx_t nb;
  const float *xb;
  const float *q;
  size_t ndis;

  float operator () (idx_t i) override {
    ndis++;
    return -faiss::fvec_inner_product(q, xb + i * d, d);
  }

  float symmetric_dis(idx_t i, idx_t j) override {
    return -faiss::fvec_inner_product(xb + j * d, xb + i * d, d);
  }

  explicit FlatIPDis(size_t d, idx_t nb, 
                    const float *xb = nullptr,
                    const float *q = nullptr)
      : d(d),
        nb(nb),
        xb(xb),
        q(q),
        ndis(0) {}

  void set_query(const float *x) override {
      q = x;
  }
};

};

DistanceComputer * GammaHNSWIndex::GetDistanceComputer() const {
  if (metric_type == faiss::METRIC_L2) {
      return new FlatL2Dis(d, indexed_vec_count_, raw_vec_head_);
  } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
      return new FlatIPDis(d, indexed_vec_count_, raw_vec_head_);
  } else {
    return nullptr;
  }
}

int GammaHNSWIndex::AddVertices(size_t n0, size_t n, 
  const float *x, bool verbose, bool preset_levels) {

#ifdef PERFORMANCE_TESTING
  double t0 = utils::getmillisecs();
#endif // PERFORMANCE_TESTING
  if (n == 0) {
    return 0;
  }
  pthread_rwlock_wrlock(&mutex_);  
  gamma_hnsw_.prepare_level_tab(n, preset_levels);
  pthread_rwlock_unlock(&mutex_);

  // add vectors from highest to lowest level
  std::vector<int> hist;
  std::vector<int> order(n);

  //add lock for each node
  std::vector<omp_lock_t> tmp_locks(n);
  locks_.insert(locks_.end(), tmp_locks.data(), tmp_locks.data() + n);

  for(size_t i = 0; i < n; i++) {    
    omp_init_lock(&locks_[i + n0]);
  }

  { // make buckets with vectors of the same level

    // build histogram
    for (size_t i = 0; i < n; i++) {
      storage_idx_t pt_id = i + n0;
      size_t pt_level = gamma_hnsw_.levels[pt_id] - 1;
      while (pt_level >= hist.size())
          hist.push_back(0);
      hist[pt_level] ++;
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

  { // perform add
    RandomGenerator rng2(789);

    int i1 = n;

    for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
      int i0 = i1 - hist[pt_level];

      if (verbose) {
        LOG(INFO) << "Adding " << i1 - i0 << " elements at level "
          << pt_level;
      }

      // random permutation to get rid of dataset order bias
      for (int j = i0; j < i1; j++)
        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
      
      int num_threads = (i1-i0) < omp_get_max_threads() ? (i1-i0) : 
                          omp_get_max_threads();

#pragma omp parallel if(i1 > i0 + 10) num_threads(num_threads)
      {
        DistanceComputer *dis = GetDistanceComputer();
        faiss::ScopeDeleter1<DistanceComputer> del(dis);
        
#pragma omp for schedule(dynamic)
        for (int i = i0; i < i1; i++) {
          storage_idx_t pt_id = order[i];
          dis->set_query (x + (pt_id - n0) * d);
          gamma_hnsw_.AddWithLocks(*dis, pt_level, pt_id, locks_);
        }
      }
      i1 = i0;
    }
    FAISS_ASSERT(i1 == 0);
  }

#ifdef PERFORMANCE_TESTING
  add_count_ += n;
  if(add_count_ >= 10000) {
    LOG(INFO) << "adding elements on top of " << n0
              << ", average add time " << (utils::getmillisecs() - t0) / n << " ms";
    add_count_ = 0;
  }
#endif // PERFORMANCE_TESTING
  
  return 0;
}

int GammaHNSWIndex::SearchHNSW(int n, const float *x, GammaSearchCondition *condition,
                   float *distances, idx_t *labels, int *total) {
  int num_threads = n < omp_get_max_threads() ? n : 
                          omp_get_max_threads();
                          
  int k = condition->topn; // topK

#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
  for (int i = 0; i < n; i ++) {
    DistanceComputer *dis = GetDistanceComputer();
    faiss::ScopeDeleter1<DistanceComputer> del(dis);
    idx_t * idxi = labels + i * k;
    float * simi = distances + i * k;
    dis->set_query(x + i * d);
    
    faiss::maxheap_heapify(k, simi, idxi);

    pthread_rwlock_rdlock(&mutex_);
    gamma_hnsw_.Search(*dis, k, idxi, simi, 
      docids_bitmap_, condition->range_query_result);
    pthread_rwlock_unlock(&mutex_);

    faiss::maxheap_reorder(k, simi, idxi);
    
    if (metric_type == faiss::METRIC_L2) {
      FlatL2Dis *l2_dis = dynamic_cast<FlatL2Dis*>(dis);
      total[i] = l2_dis->ndis;
    } else {
      FlatIPDis *ip_dis = dynamic_cast<FlatIPDis*>(dis);
      total[i] = ip_dis->ndis;
    }

    if (reconstruct_from_neighbors &&
      reconstruct_from_neighbors->k_reorder != 0) {
      int k_reorder = reconstruct_from_neighbors->k_reorder;
      if (k_reorder == -1 || k_reorder > k) k_reorder = k;

      reconstruct_from_neighbors->compute_distances(
               k_reorder, idxi, x + i * d, simi);

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
  std::string compute_msg = "compute ";
  compute_msg += std::to_string(n);
  condition->Perf(compute_msg);
#endif // PERFORMANCE_TESTING

  return 0;
}

int GammaHNSWIndex::Search(const VectorQuery *query,
                            GammaSearchCondition *condition,
                            VectorResult &result) {

  float *x = reinterpret_cast<float *>(query->value->value);
  int n = query->value->len / (d * sizeof(float));

  idx_t *labels = reinterpret_cast<idx_t *>(result.docids);

  if (condition->use_direct_search) {
    SearchDirectly(n, x, condition, result.dists, labels, result.total.data());
  } else {
    SearchHNSW(n, x, condition, result.dists, labels, result.total.data());
  }

  for (int i = 0; i < n; i++) {
    int pos = 0;

    std::map<int, int> docid2count;
    for (int j = 0; j < condition->topn; j++) {
      long *docid = result.docids + i * condition->topn + j;
      if (docid[0] == -1) continue;
      int vector_id = (int)docid[0];
      int real_docid = this->raw_vec_->vid_mgr_->VID2DocID(vector_id);
      if (docid2count.find(real_docid) == docid2count.end()) {
        int real_pos = i * condition->topn + pos;
        result.docids[real_pos] = real_docid;
        int ret = this->raw_vec_->GetSource(vector_id, result.sources[real_pos],
                                            result.source_lens[real_pos]);
        if (ret != 0) {
          result.sources[real_pos] = nullptr;
          result.source_lens[real_pos] = 0;
        }
        result.dists[real_pos] = result.dists[i * condition->topn + j];

        pos++;
        docid2count[real_docid] = 1;
      }
    }

    if (pos > 0) {
      result.idx[i] = 0;  // init start id of seeking
    }

    for (; pos < condition->topn; pos++) {
      result.docids[i * condition->topn + pos] = -1;
      result.dists[i * condition->topn + pos] = -1;
    }
  }

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

//TODO
int GammaHNSWIndex::Update(int doc_id, const float *vec) { return 0; }

int GammaHNSWIndex::Delete(int doc_id) { return 0; }

int GammaHNSWIndex::Dump(const std::string &dir, int max_vid) { return 0; }

int GammaHNSWIndex::Load(const std::vector<std::string> &index_dirs) { return 0; }

GammaHNSWFlatIndex::GammaHNSWFlatIndex(size_t d,                                 
                                DistanceMetricType metric_type,
                                int nlinks, int efSearch, int efConstruction,
                                const char *docids_bitmap, 
                                RawVector<float> *raw_vec):
  GammaHNSWIndex(new faiss::IndexFlatL2(d), d, metric_type, nlinks,
                efSearch, efConstruction, 
                docids_bitmap, raw_vec) {
  own_fields = true;
  is_trained = true;
}

}
