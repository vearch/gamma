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
#include "mmap_raw_vector.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "bitmap.h"
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

GammaIVFPQIndex::GammaIVFPQIndex(faiss::Index *quantizer, size_t d,
                                 size_t nlist, size_t M, size_t nbits_per_idx,
                                 const char *docids_bitmap,
                                 RawVector<float> *raw_vec,
                                 GammaCounters *counters)
    : GammaFLATIndex(d, docids_bitmap, raw_vec),
      faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx),
      indexed_vec_count_(0),
      gamma_counters_(counters) {
  assert(raw_vec != nullptr);
  int max_vec_size = raw_vec->GetMaxVectorSize(); 

  this->SetRawVectorFloat(raw_vec);
  rt_invert_index_ptr_ = new realtime::RTInvertIndex(
      this->nlist, this->code_size, max_vec_size, raw_vec_->vid_mgr_,
      docids_bitmap, 100000, 12800000);

  if (this->invlists) {
    delete this->invlists;
    this->invlists = nullptr;
  }

  bool ret = rt_invert_index_ptr_->Init();

  if (ret) {
    this->invlists =
        new realtime::RTInvertedLists(rt_invert_index_ptr_, nlist, code_size);
  }

  // default value, nprobe will be passed at search time
  this->nprobe = 20;

  compaction_ = false;
  compact_bucket_no_ = 0;
  compacted_num_ = 0;
  updated_num_ = 0;

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
}

faiss::InvertedListScanner *GammaIVFPQIndex::get_InvertedListScanner(
    bool store_pairs) const {
  return GetGammaInvertedListScanner(store_pairs);
}

GammaInvertedListScanner *GammaIVFPQIndex::GetGammaIVFFlatScanner
     (size_t d) const
{
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    auto scanner = new GammaIVFFlatScanner<
        faiss::METRIC_INNER_PRODUCT, faiss::CMin<float, int64_t> > (d);
    scanner->SetVecFilter(this->docids_bitmap_, this->raw_vec_);
    return scanner;
  } else if (metric_type == faiss::METRIC_L2) {
    auto scanner = new GammaIVFFlatScanner<
        faiss::METRIC_L2, faiss::CMax<float, int64_t> >(d);
    scanner->SetVecFilter(this->docids_bitmap_, this->raw_vec_);
    return scanner;
  } else {
    LOG(ERROR) << "metric type not supported";
  }
  return nullptr;
}

GammaInvertedListScanner *GammaIVFPQIndex::GetGammaInvertedListScanner(
    bool store_pairs) const {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    auto scanner =
        new GammaIVFPQScanner<faiss::METRIC_INNER_PRODUCT,
                              faiss::CMin<float, idx_t>, 2>(*this, store_pairs);
    scanner->SetVecFilter(this->docids_bitmap_, this->raw_vec_);
    return scanner;
  } else if (metric_type == faiss::METRIC_L2) {
    auto scanner =
        new GammaIVFPQScanner<faiss::METRIC_L2, faiss::CMax<float, idx_t>, 2>(
            *this, store_pairs);
    scanner->SetVecFilter(this->docids_bitmap_, this->raw_vec_);
    return scanner;
  }
  return nullptr;
}

int GammaIVFPQIndex::Indexing() {
  if (this->is_trained) {
    LOG(INFO) << "gamma ivfpq index is already trained, skip indexing";
    return 0;
  }
  int vectors_count = raw_vec_->GetVectorNum();
  if (vectors_count < 8192) {
    LOG(ERROR) << "vector total count [" << vectors_count
               << "] less then 8192, failed!";
    return -1;
  }
  size_t num = vectors_count > 100000 ? 100000 : vectors_count;
  ScopeVector<float> header;
  raw_vec_->GetVectorHeader(0, num, header);

  int raw_d = raw_vec_->GetDimension();

  float *train_vec = nullptr;

  if (d_ > raw_d) {
    float *vec = new float[num * d_];

    ConvertVectorDim(num, raw_d, d, header.Get(), vec);

    train_vec = vec;
  } else {
    train_vec = const_cast<float *>(header.Get());
  }

  train(num, train_vec);

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

int GammaIVFPQIndex::Delete(int docid) {
  std::vector<int> vids;
  raw_vec_->vid_mgr_->DocID2VID(docid, vids);
  rt_invert_index_ptr_->Delete(vids.data(), vids.size());
  return 0;
}

int GammaIVFPQIndex::AddRTVecsToIndex() {
  int ret = 0;
  int total_stored_vecs = raw_vec_->GetVectorNum();
  if (indexed_vec_count_ > total_stored_vecs) {
    LOG(ERROR) << "internal error : indexed_vec_count=" << indexed_vec_count_
               << " should not greater than total_stored_vecs="
               << total_stored_vecs;
    ret = -1;
  } else if (indexed_vec_count_ == total_stored_vecs) {
#ifdef DEBUG
    LOG(INFO) << "no extra vectors existed for indexing";
#endif
    rt_invert_index_ptr_->CompactIfNeed();
  } else {
    int MAX_NUM_PER_INDEX = 1000;
    int index_count =
        (total_stored_vecs - indexed_vec_count_) / MAX_NUM_PER_INDEX + 1;

    for (int i = 0; i < index_count; i++) {
      int start_docid = indexed_vec_count_;
      size_t count_per_index =
          (i == (index_count - 1) ? total_stored_vecs - start_docid
                                  : MAX_NUM_PER_INDEX);
      ScopeVector<float> vector_head;
      raw_vec_->GetVectorHeader(indexed_vec_count_,
                                indexed_vec_count_ + count_per_index,
                                vector_head);

      int raw_d = raw_vec_->GetDimension();
      float *add_vec = nullptr;

      if (d_ > raw_d) {
        float *vec = new float[count_per_index * d_];

        ConvertVectorDim(count_per_index, raw_d, d, vector_head.Get(), vec);
        add_vec = vec;
      } else {
        add_vec = const_cast<float *>(vector_head.Get());
      }

      if (!Add(count_per_index, add_vec)) {
        LOG(ERROR) << "add index from docid " << start_docid << " error!";
        ret = -2;
      }

      if (d_ > raw_d) {
        delete add_vec;
      }
    }
  }
  if (AddUpdatedVecToIndex()) {
    LOG(ERROR) << "add updated vectors to index error";
    return -1;
  }
  return ret;
}  // namespace tig_gamma

int GammaIVFPQIndex::AddUpdatedVecToIndex() {
  std::vector<long> vids;
  int vid;
  while (raw_vec_->updated_vids_->try_dequeue(vid)) {
    if (bitmap::test(docids_bitmap_, raw_vec_->vid_mgr_->VID2DocID(vid)))
      continue;
    vids.push_back(vid);
    if (vids.size() >= 20000) break;
  }
  if (vids.size() == 0) return 0;
  ScopeVectors<float> scope_vecs(vids.size());
  raw_vec_->Gets(vids.size(), vids.data(), scope_vecs);
  int raw_d = raw_vec_->GetDimension();
  for (size_t i = 0; i < vids.size(); i++) {
    const float *vec = nullptr;
    ScopeVector<float> del;
    if (d_ > raw_d) {
      float *extend_vec = new float[d_];
      ConvertVectorDim(1, raw_d, d_, scope_vecs.Get(i), extend_vec);
      vec = (const float *)extend_vec;
      del.Set(extend_vec);
    } else {
      vec = scope_vecs.Get(i);
    }
    
    idx_t idx = -1;
    quantizer->assign(1, vec, &idx);

    std::vector<uint8_t> xcodes;
    xcodes.resize(code_size);
    const float *to_encode = nullptr;
    faiss::ScopeDeleter<float> del_to_encode;

    if (by_residual) {
      to_encode = compute_residuals(quantizer, 1, vec, &idx);
      del_to_encode.set(to_encode);
    } else {
      to_encode = vec;
    }
    pq.compute_codes(to_encode, xcodes.data(), 1);
    rt_invert_index_ptr_->Update(idx, vids[i], xcodes);
  }
  updated_num_ += vids.size();
  LOG(INFO) << "update index success! size=" << vids.size()
            << ", total=" << updated_num_;
  return 0;
}

bool GammaIVFPQIndex::Add(int n, const float *vec) {
#ifdef PERFORMANCE_TESTING
  double t0 = faiss::getmillisecs();
#endif
  std::map<int, std::vector<long>> new_keys;
  std::map<int, std::vector<uint8_t>> new_codes;

  idx_t *idx;
  faiss::ScopeDeleter<idx_t> del_idx;

  idx_t *idx0 = new idx_t[n];
  quantizer->assign(n, vec, idx0);
  idx = idx0;
  del_idx.set(idx);

  uint8_t *xcodes = new uint8_t[n * code_size];
  faiss::ScopeDeleter<uint8_t> del_xcodes(xcodes);

  const float *to_encode = nullptr;
  faiss::ScopeDeleter<float> del_to_encode;

  if (by_residual) {
    to_encode = compute_residuals(quantizer, n, vec, idx);
    del_to_encode.set(to_encode);
  } else {
    to_encode = vec;
  }
  pq.compute_codes(to_encode, xcodes, n);

  size_t n_ignore = 0;
  long vid = indexed_vec_count_;
  for (int i = 0; i < n; i++) {
    long key = idx[i];
    assert(key < (long)nlist);
    if (key < 0) {
      n_ignore++;
      continue;
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
  if (add_count_ >= 100000) {
    double t1 = faiss::getmillisecs();
    LOG(INFO) << "Add time [" << (t1 - t0) / n << "]ms, count "
              << indexed_vec_count_;
    // rt_invert_index_ptr_->PrintBucketSize();
    add_count_ = 0;
  }
#endif
  return true;
}

void GammaIVFPQIndex::SearchIVFPQ(int n, const float *x,
                                  GammaSearchCondition *condition,
                                  float *distances, idx_t *labels, int *total) {
  size_t nprobe = this->nprobe;
  if (condition->nprobe > 0 && (size_t)condition->nprobe <= this->nlist) {
    nprobe = condition->nprobe;
  } else {
    LOG(WARNING) << "Error nprobe for search, so using default value:" << this->nprobe;
    condition->nprobe = nprobe;
  }

  std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
  std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

  quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

  this->invlists->prefetch_lists(idx.get(), n * nprobe);

  if(condition->ivf_flat)
    search_ivf_flat(n, x, condition, idx.get(), coarse_dis.get(), distances, 
                    labels, total, false);
  else
    search_preassigned(n, x, condition, idx.get(), coarse_dis.get(), distances,
                     labels, total, false);
}

namespace {

using HeapForIP = faiss::CMin<float, idx_t>;
using HeapForL2 = faiss::CMax<float, idx_t>;

// intialize + reorder a result heap

int init_result(faiss::MetricType metric_type, int k, float *simi, idx_t *idxi) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    faiss::heap_heapify<HeapForIP> (k, simi, idxi);
  } else {
    faiss::heap_heapify<HeapForL2> (k, simi, idxi);
  }
  return 0;
};

int reorder_result(faiss::MetricType metric_type, int k, float *simi, idx_t *idxi) {
  if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    faiss::heap_reorder<HeapForIP> (k, simi, idxi);
  } else {
    faiss::heap_reorder<HeapForL2> (k, simi, idxi);
  }
  return 0;
};

// single list scan using the current scanner (with query
// set porperly) and storing results in simi and idxi
size_t scan_one_list(GammaInvertedListScanner *scanner, idx_t key, 
    float coarse_dis_i, float *simi, idx_t *idxi, int k, 
    idx_t nlist, faiss::InvertedLists *invlists, 
    bool store_pairs, bool ivf_flat, float *raw_vec_head = nullptr) {
  if (key < 0) {
    // not enough centroids for multiprobe
    return 0;
  }
  if (key >= (idx_t) nlist) {
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
    sids.reset (new faiss::InvertedLists::ScopedIds (invlists, key));
    ids = sids->get();
  }

  scanner->set_list(key, coarse_dis_i);
  
  //scan_codes need uint8_t *
  const uint8_t *codes = nullptr;
  
  if(ivf_flat) {
    codes = reinterpret_cast<uint8_t *>(raw_vec_head);
  } else {
    faiss::InvertedLists::ScopedCodes scodes(invlists, key);
    codes = scodes.get(); 
  }
  scanner->scan_codes(list_size, codes, ids, simi, idxi, k);

  return list_size;
};

}

void GammaIVFPQIndex::search_ivf_flat(
    int n, const float *x, GammaSearchCondition *condition, const idx_t *keys, 
    const float *coarse_dis, float *distances, idx_t *labels, int *total, 
    bool store_pairs, const faiss::IVFSearchParameters *params) {
  int k = condition->topn; // topk
  if (k <= 0) {
    LOG(WARNING) << "topK should greater then 0, topK = " << k;
    return;
  }

  auto raw_vec_type = dynamic_cast<MmapRawVector<float> *>(raw_vec_);
  if(raw_vec_type == nullptr || raw_vec_type->GetMemoryMode() == 0) {
    LOG(WARNING) << "IVF FLAT cann't work in RocksDB or in disk mode";
    memset(labels, -1, n * sizeof(idx_t) * k);
    return;
  }

  ScopeVector<float> vector_head;
  int vector_num = raw_vec_->GetVectorNum();
  float *raw_vec_head = nullptr;
  if(raw_vec_->GetVectorHeader(0, vector_num, vector_head)) {
    LOG(ERROR) << "Cann't get raw_vec head";
    memset(labels, -1, n * sizeof(idx_t) * k);
    return;
  } else {
    raw_vec_head = const_cast<float *>(vector_head.Get());
  }

  size_t raw_d = raw_vec_->GetDimension();

  size_t nprobe = condition->nprobe;
  
  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  condition->parallel_mode = condition->parallel_based_on_query ? 0 : 1;
  int ni_total = -1;
  if (condition->range_query_result &&
      condition->range_query_result->GetAllResult() != nullptr) {
    ni_total = condition->range_query_result->GetAllResult()->Size();
  }

  // don't start parallel section if single query
  bool do_parallel = condition->parallel_mode == 0 ? n > 1 : nprobe > 1;

  size_t ndis = 0;
#pragma omp parallel if(do_parallel) reduction(+: ndis)
  {
    GammaInvertedListScanner *scanner = GetGammaIVFFlatScanner(raw_d);
    faiss::ScopeDeleter1<GammaInvertedListScanner> del(scanner);
    scanner->set_search_condition(condition);

    /****************************************************
    * Actual loops, depending on parallel_mode
    ****************************************************/
  
    if (condition->parallel_mode == 0) { // parallelize over queries
  
#pragma omp for
      for (int i = 0; i < n; i++) {
  
        // loop over queries
        scanner->set_query (x + i * d);
        float * simi = distances + i * k;
        idx_t * idxi = labels + i * k;
  
        init_result (metric_type, k, simi, idxi);
  
        size_t nscan = 0;
  
        // loop over probes
        for (size_t ik = 0; ik < nprobe; ik++) {
          nscan += scan_one_list (
              scanner, keys [i * nprobe + ik], coarse_dis[i * nprobe + ik], 
              simi, idxi, k, this->nlist, this->invlists, store_pairs, 
              condition->ivf_flat, raw_vec_head);
  
          if (max_codes && nscan >= max_codes) {
            break;
          }
        }
        total[i] = ni_total;
  
        ndis += nscan;
        reorder_result (metric_type, k, simi, idxi);
      } // parallel for
    } else { // parallelize over inverted lists
      std::vector <idx_t> local_idx (k);
      std::vector <float> local_dis (k);
  
      for (int i = 0; i < n; i++) {
        scanner->set_query (x + i * d);
        init_result (metric_type, k, local_dis.data(), local_idx.data());
  
#pragma omp for schedule(dynamic)
        for (size_t ik = 0; ik < nprobe; ik++) {
          ndis += scan_one_list (
              scanner, keys [i * nprobe + ik], coarse_dis[i * nprobe + ik], 
              local_dis.data(), local_idx.data(), k, this->nlist, 
              this->invlists, store_pairs, condition->ivf_flat, raw_vec_head);
  
            // can't do the test on max_codes
        }
  
        total[i] = ni_total;
  
        // merge thread-local results
  
        float * simi = distances + i * k;
        idx_t * idxi = labels + i * k;
#pragma omp single
        init_result (metric_type, k, simi, idxi);
  
#pragma omp barrier
#pragma omp critical
        {
          if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            faiss::heap_addn<HeapForIP>
              (k, simi, idxi, local_dis.data(), local_idx.data(), k);
          } else {
            faiss::heap_addn<HeapForL2>
              (k, simi, idxi, local_dis.data(), local_idx.data(), k);
          }
        }
#pragma omp barrier
#pragma omp single
        reorder_result (metric_type, k, simi, idxi);
      }
    }
  } // parallel section
#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "ivf flat compute ";
  compute_msg += std::to_string(n);
  condition->Perf(compute_msg);
#endif
}

void GammaIVFPQIndex::search_preassigned(
    int n, const float *x, GammaSearchCondition *condition, const idx_t *keys,
    const float *coarse_dis, float *distances, idx_t *labels, int *total,
    bool store_pairs, const faiss::IVFSearchParameters *params) {
  int nprobe = condition->nprobe;

  long max_codes = params ? params->max_codes : this->max_codes;

  long k = condition->topn;  // topK
  if (k <= 0) {
    LOG(WARNING) << "topK is should greater then 0, topK = " << k;
    return;
  }
  size_t ndis = 0;

  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  const int recall_num = condition->recall_num;
  float *recall_distances = new float[n * recall_num];
  idx_t *recall_labels = new idx_t[n * recall_num];
  faiss::ScopeDeleter<float> del1(recall_distances);
  faiss::ScopeDeleter<idx_t> del2(recall_labels);

  std::function<void(const float *, float *, idx_t *, float *, idx_t *)>
      compute_dis;

  if (condition->has_rank) {
    // calculate inner product for selected possible vectors
    compute_dis = [&](const float *xi, float *simi, idx_t *idxi,
                      float *recall_simi, idx_t *recall_idxi) {
      ScopeVectors<float> scope_vecs(recall_num);
      raw_vec_->Gets(recall_num, (long *)recall_idxi, scope_vecs);
      const float **vecs = scope_vecs.Get();
      int raw_d = raw_vec_->GetDimension();
      for (int j = 0; j < recall_num; j++) {
        if (recall_idxi[j] == -1) continue;
        float dis = 0;
        if (metric_type == faiss::METRIC_INNER_PRODUCT) {
          dis = faiss::fvec_inner_product(xi, vecs[j], raw_d);
        } else {
          dis = faiss::fvec_L2sqr(xi, vecs[j], raw_d);
        }

        if (((condition->min_dist >= 0 && dis >= condition->min_dist) &&
             (condition->max_dist >= 0 && dis <= condition->max_dist)) ||
            (condition->min_dist == -1 && condition->max_dist == -1)) {
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
      if (condition->sort_by_docid) {  // sort by doc id
        std::vector<std::pair<idx_t, float>> id_sim_pairs;
        for (int i = 0; i < k; i++) {
          id_sim_pairs.emplace_back(std::make_pair(idxi[i], simi[i]));
        }
        std::sort(id_sim_pairs.begin(), id_sim_pairs.end());
        for (int i = 0; i < k; i++) {
          idxi[i] = id_sim_pairs[i].first;
          simi[i] = id_sim_pairs[i].second;
        }
      } else {  // sort by distance
        reorder_result(metric_type, k, simi, idxi);
      }
    };
  } else {
    // compute without rank
    compute_dis = [&](const float *xi, float *simi, idx_t *idxi,
                      float *recall_simi, idx_t *recall_idxi) {
      int i = 0;
      for (int j = 0; j < recall_num; j++) {
        if (recall_idxi[j] == -1) continue;
        float dis = recall_simi[j];

        if (((condition->min_dist >= 0 && dis >= condition->min_dist) &&
             (condition->max_dist >= 0 && dis <= condition->max_dist)) ||
            (condition->min_dist == -1 && condition->max_dist == -1)) {
          simi[i] = dis;
          idxi[i] = recall_idxi[j];
          ++i;
        }
      }
      reorder_result(metric_type, k, simi, idxi);
    };
  }
#ifdef PERFORMANCE_TESTING
  condition->Perf("search prepare");
#endif

  condition->parallel_mode = condition->parallel_based_on_query ? 0 : 1;
  int ni_total = -1;
  if (condition->range_query_result &&
      condition->range_query_result->GetAllResult() != nullptr) {
    ni_total = condition->range_query_result->GetAllResult()->Size();
  }

  // don't start parallel section if single query
  bool do_parallel = condition->parallel_mode == 0 ? n > 1 : nprobe > 1;

#ifdef SMALL_DOC_NUM_OPTIMIZATION
  double s_start = utils::getmillisecs();
  if (condition->range_query_result &&
      condition->range_query_result->GetAllResult() != nullptr &&
      condition->range_query_result->GetAllResult()->Size() < 50000) {
    const std::vector<int> docid_list = condition->range_query_result->ToDocs();

#ifdef DEBUG
    size_t docid_size = docid_list.size();
    LOG(INFO) << utils::join(docid_list.data(),
                             docid_size > 1000 ? 1000 : docid_size, ',');
#endif

    std::vector<int> vid_list(docid_list.size() * MAX_VECTOR_NUM_PER_DOC);
    int *vid_list_data = vid_list.data();
    int *curr_ptr = vid_list_data;
    for (size_t i = 0; i < docid_list.size(); i++) {
      if (bitmap::test(this->docids_bitmap_, docid_list[i])) {
        continue;
      }

      int *vids = this->raw_vec_->docid2vid_[docid_list[i]];
      if (vids) {
        memcpy((void *)(curr_ptr), (void *)(vids + 1), sizeof(int) * vids[0]);
        curr_ptr += vids[0];
      }
    }
    int vid_list_len = curr_ptr - vid_list_data;

#ifdef PERFORMANCE_TESTING
    double to_vid_end = utils::getmillisecs();
#endif

    std::vector<std::vector<const uint8_t *>> bucket_codes;
    std::vector<std::vector<long>> bucket_vids;
    int ret = ((realtime::RTInvertedLists *)this->invlists)
                  ->rt_invert_index_ptr_->RetrieveCodes(
                      vid_list_data, vid_list_len, bucket_codes, bucket_vids);
    if (ret != 0) throw std::runtime_error("retrieve codes by vid error");

#ifdef PERFORMANCE_TESTING
    double retrieve_code_end = utils::getmillisecs();
#endif

#pragma omp parallel if (do_parallel) reduction(+ : ndis)
    {
      GammaInvertedListScanner *scanner =
          GetGammaInvertedListScanner(store_pairs);
      faiss::ScopeDeleter1<GammaInvertedListScanner> del(scanner);
      scanner->set_search_condition(condition);

#pragma omp for
      for (int i = 0; i < n; i++) {  // loop over queries

#ifdef PERFORMANCE_TESTING
        double query_start = utils::getmillisecs();
#endif

        const float *xi = x + i * d;
        scanner->set_query(x + i * d);

        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        float *recall_simi = recall_distances + i * recall_num;
        idx_t *recall_idxi = recall_labels + i * recall_num;

        init_result(metric_type, k, simi, idxi);
        init_result(metric_type, recall_num, recall_simi, recall_idxi);

        for (int ik = 0; ik < nprobe; ik++) {
          long key = keys[i * nprobe + ik];
          float coarse_dis_i = coarse_dis[i * nprobe + ik];
          size_t ncode = bucket_codes[key].size();
          if (ncode <= 0) {
            continue;
          }
          const uint8_t **codes = bucket_codes[key].data();
          const idx_t *vids =
              reinterpret_cast<idx_t *>(bucket_vids[key].data());

          scanner->set_list(key, coarse_dis_i);
          scanner->scan_codes_pointer(ncode, codes, vids, recall_simi,
                                      recall_idxi, recall_num);
        }

#ifdef PERFORMANCE_TESTING
        double coarse_end = utils::getmillisecs();
#endif
        compute_dis(xi, simi, idxi, recall_simi, recall_idxi);

        total[i] = ni_total;

#ifdef PERFORMANCE_TESTING
        if (++search_count_ % 1000 == 0) {
          double end = utils::getmillisecs();
          LOG(INFO) << "ivfqp range filter, doc id list size="
                    << docid_list.size() << ", vid list len=" << vid_list_len
                    << "to docid cost=" << to_vid_end - s_start
                    << "ms, retrieve code cost="
                    << retrieve_code_end - to_vid_end
                    << "ms, query[coarse cost=" << coarse_end - query_start
                    << "ms, reorder cost=" << end - coarse_end
                    << "ms, total cost=" << end - s_start
                    << "ms] metric type=" << metric_type
                    << ", nprobe=" << this->nprobe;
        }
#endif
      }
    }
    return;
  }
#endif  // SMALL_DOC_NUM_OPTIMIZATION

#pragma omp parallel if (do_parallel) reduction(+ : ndis)
  {
    GammaInvertedListScanner *scanner =
        GetGammaInvertedListScanner(store_pairs);
    faiss::ScopeDeleter1<GammaInvertedListScanner> del(scanner);
    scanner->set_search_condition(condition);

    if (condition->parallel_mode == 0) {  // parallelize over queries
#pragma omp for
      for (int i = 0; i < n; i++) {
        // loop over queries
        const float *xi = x + i * d;
        scanner->set_query(x + i * d);
        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        float *recall_simi = recall_distances + i * recall_num;
        idx_t *recall_idxi = recall_labels + i * recall_num;

        init_result(metric_type, k, simi, idxi);
        init_result(metric_type, recall_num, recall_simi, recall_idxi);

        long nscan = 0;

        // loop over probes
        for (int ik = 0; ik < nprobe; ik++) {
          nscan +=
              scan_one_list(scanner, keys[i * nprobe + ik], 
                  coarse_dis[i * nprobe + ik], recall_simi, 
                  recall_idxi, recall_num, this->nlist,
                  this->invlists, store_pairs, condition->ivf_flat);

          if (max_codes && nscan >= max_codes) break;
        }
        total[i] = ni_total;

        ndis += nscan;
        compute_dis(xi, simi, idxi, recall_simi, recall_idxi);
      }       // parallel for
    } else {  // parallelize over inverted lists
      std::vector<idx_t> local_idx(recall_num);
      std::vector<float> local_dis(recall_num);

      for (int i = 0; i < n; i++) {
        const float *xi = x + i * d;
        scanner->set_query(xi);

        init_result(metric_type, recall_num, local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
        for (int ik = 0; ik < nprobe; ik++) {
          ndis +=
              scan_one_list(scanner, keys[i * nprobe + ik], 
                  coarse_dis[i * nprobe + ik], local_dis.data(), 
                  local_idx.data(), recall_num, this->nlist,
                  this->invlists, store_pairs, condition->ivf_flat);

          // can't do the test on max_codes
        }

        total[i] = ni_total;

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
          condition->Perf("coarse");
#endif
          compute_dis(xi, simi, idxi, recall_simi, recall_idxi);

#ifdef PERFORMANCE_TESTING
          condition->Perf("reorder");
#endif
        }
      }
    }
  }  // parallel

#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "compute ";
  compute_msg += std::to_string(n);
  condition->Perf(compute_msg);
#endif
}

int GammaIVFPQIndex::Search(const VectorQuery *query,
                            GammaSearchCondition *condition,
                            VectorResult &result) {
  float *x = reinterpret_cast<float *>(query->value->value);
  int raw_d = raw_vec_->GetDimension();
  size_t n = query->value->len / (raw_d * sizeof(float));

  if (condition->metric_type == InnerProduct) {
    metric_type = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type = faiss::METRIC_L2;
  }
  idx_t *idx = reinterpret_cast<idx_t *>(result.docids);

  float *vec_q = nullptr;

  if (d > raw_d) {
    float *vec = new float[n * d];

    ConvertVectorDim(n, raw_d, d, x, vec);

    vec_q = vec;
  } else {
    vec_q = x;
  }

  if (condition->use_direct_search) {
    SearchDirectly(n, vec_q, condition, result.dists, idx, result.total.data());
  } else {
    SearchIVFPQ(n, vec_q, condition, result.dists, idx, result.total.data());
  }

  if (d > raw_d) {
    delete vec_q;
  }

  for (size_t i = 0; i < n; i++) {
    int pos = 0;

    std::map<int, int> docid2count;
    for (int j = 0; j < condition->topn; j++) {
      long *docid = result.docids + i * condition->topn + j;
      if (docid[0] == -1) continue;
      int vector_id = (int)docid[0];
      int real_docid = raw_vec_->vid_mgr_->VID2DocID(vector_id);
      if (docid2count.find(real_docid) == docid2count.end()) {
        int real_pos = i * condition->topn + pos;
        result.docids[real_pos] = real_docid;
        int ret = raw_vec_->GetSource(vector_id, result.sources[real_pos],
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

  /*************************************************************
   * I/O macros
   *
   * we use macros so that we have a line number to report in abort
   * (). This makes debugging a lot easier. The IOReader or IOWriter is
   * always called f and thus is not passed in as a macro parameter.
   **************************************************************/

#define WRITEANDCHECK(ptr, n)                                                 \
  {                                                                           \
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                \
    FAISS_THROW_IF_NOT_FMT(ret == (n), "write error in %s: %ld != %ld (%s)",  \
                           f->name.c_str(), ret, size_t(n), strerror(errno)); \
  }

#define READANDCHECK(ptr, n)                                                  \
  {                                                                           \
    size_t ret = (*f)(ptr, sizeof(*(ptr)), n);                                \
    FAISS_THROW_IF_NOT_FMT(ret == (n), "read error in %s: %ld != %ld (%s)",   \
                           f->name.c_str(), ret, size_t(n), strerror(errno)); \
  }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)
#define READ1(x) READANDCHECK(&(x), 1)

#define WRITEVECTOR(vec)               \
  {                                    \
    size_t size = (vec).size();        \
    WRITEANDCHECK(&size, 1);           \
    WRITEANDCHECK((vec).data(), size); \
  }

// will fail if we write 256G of data at once...
#define READVECTOR(vec)                                 \
  {                                                     \
    size_t size;                                        \
    READANDCHECK(&size, 1);                             \
    FAISS_THROW_IF_NOT(size >= 0 && size < (1L << 40)); \
    (vec).resize(size);                                 \
    READANDCHECK((vec).data(), size);                   \
  }

/****************************************************************
 * Write
 *****************************************************************/
static void write_index_header(const faiss::Index *idx, faiss::IOWriter *f) {
  WRITE1(idx->d);
  WRITE1(idx->ntotal);
  faiss::Index::idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx->is_trained);
  WRITE1(idx->metric_type);
}

static void write_direct_map(const faiss::DirectMap *dm, faiss::IOWriter *f) {
  char maintain_direct_map =
      (char)dm->type;  // for backwards compatibility with bool
  WRITE1(maintain_direct_map);
  WRITEVECTOR(dm->array);
  if (dm->type == faiss::DirectMap::Hashtable) {
    using idx_t = faiss::Index::idx_t;
    std::vector<std::pair<idx_t, idx_t>> v;
    const std::unordered_map<idx_t, idx_t> &map = dm->hashtable;
    v.resize(map.size());
    std::copy(map.begin(), map.end(), v.begin());
    WRITEVECTOR(v);
  }
}

static void write_ivf_header(const faiss::IndexIVF *ivf, faiss::IOWriter *f) {
  write_index_header(ivf, f);
  WRITE1(ivf->nlist);
  WRITE1(ivf->nprobe);
  faiss::write_index(ivf->quantizer, f);
  write_direct_map(&ivf->direct_map, f);
}

static void read_index_header(faiss::Index *idx, faiss::IOReader *f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  faiss::Index::idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  idx->verbose = false;
}

static void read_direct_map(faiss::DirectMap *dm, faiss::IOReader *f) {
  char maintain_direct_map;
  READ1(maintain_direct_map);
  dm->type = (faiss::DirectMap::Type)maintain_direct_map;
  READVECTOR(dm->array);
  if (dm->type == faiss::DirectMap::Hashtable) {
    using idx_t = faiss::Index::idx_t;
    std::vector<std::pair<idx_t, idx_t>> v;
    READVECTOR(v);
    std::unordered_map<idx_t, idx_t> &map = dm->hashtable;
    map.reserve(v.size());
    for (auto it : v) {
      map[it.first] = it.second;
    }
  }
}

static void read_ivf_header(
    faiss::IndexIVF *ivf, faiss::IOReader *f,
    std::vector<std::vector<faiss::Index::idx_t>> *ids = nullptr) {
  read_index_header(ivf, f);
  READ1(ivf->nlist);
  READ1(ivf->nprobe);
  ivf->quantizer = faiss::read_index(f);
  ivf->own_fields = true;
  if (ids) {  // used in legacy "Iv" formats
    ids->resize(ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++) READVECTOR((*ids)[i]);
  }
  read_direct_map(&ivf->direct_map, f);
  // READ1(ivf->maintain_direct_map);
  // READVECTOR(ivf->direct_map);
}

static void write_ProductQuantizer(const faiss::ProductQuantizer *pq,
                                   faiss::IOWriter *f) {
  WRITE1(pq->d);
  WRITE1(pq->M);
  WRITE1(pq->nbits);
  WRITEVECTOR(pq->centroids);
}

static void read_ProductQuantizer(faiss::ProductQuantizer *pq,
                                  faiss::IOReader *f) {
  READ1(pq->d);
  READ1(pq->M);
  READ1(pq->nbits);
  pq->set_derived_values();
  READVECTOR(pq->centroids);
}

struct FileIOReader : faiss::IOReader {
  FILE *f = nullptr;
  bool need_close = false;

  FileIOReader(FILE *rf) : f(rf) {}

  FileIOReader(const char *fname) {
    name = fname;
    f = fopen(fname, "rb");
    FAISS_THROW_IF_NOT_FMT(f, "could not open %s for reading: %s", fname,
                           strerror(errno));
    need_close = true;
  }

  ~FileIOReader() override {
    if (need_close) {
      int ret = fclose(f);
      if (ret != 0) {  // we cannot raise and exception in the destructor
        fprintf(stderr, "file %s close error: %s", name.c_str(),
                strerror(errno));
      }
    }
  }

  size_t operator()(void *ptr, size_t size, size_t nitems) override {
    return fread(ptr, size, nitems, f);
  }

  int fileno() override { return ::fileno(f); }
};

struct FileIOWriter : faiss::IOWriter {
  FILE *f = nullptr;
  bool need_close = false;

  FileIOWriter(FILE *wf) : f(wf) {}

  FileIOWriter(const char *fname) {
    name = fname;
    f = fopen(fname, "wb");
    FAISS_THROW_IF_NOT_FMT(f, "could not open %s for writing: %s", fname,
                           strerror(errno));
    need_close = true;
  }

  ~FileIOWriter() override {
    if (need_close) {
      int ret = fclose(f);
      if (ret != 0) {
        // we cannot raise and exception in the destructor
        fprintf(stderr, "file %s close error: %s", name.c_str(),
                strerror(errno));
      }
    }
  }

  size_t operator()(const void *ptr, size_t size, size_t nitems) override {
    return fwrite(ptr, size, nitems, f);
  }
  int fileno() override { return ::fileno(f); }
};

int GammaIVFPQIndex::Dump(const std::string &dir, int max_vid) {
  if (!rt_invert_index_ptr_) {
    LOG(INFO) << "realtime invert index ptr is null";
    return -1;
  }
  if (!this->is_trained) {
    LOG(INFO) << "gamma index is not trained, skip dumping";
    return 0;
  }
  string vec_name = raw_vec_->GetName();
  string info_file = dir + "/" + vec_name + ".index.param";
  faiss::IOWriter *f = new FileIOWriter(info_file.c_str());
  const IndexIVFPQ *ivpq = static_cast<const IndexIVFPQ *>(this);
  write_ivf_header(ivpq, f);
  WRITE1(ivpq->by_residual);
  WRITE1(ivpq->code_size);
  tig_gamma::write_ProductQuantizer(&ivpq->pq, f);
  delete f;

  LOG(INFO) << "dump: d=" << ivpq->d << ", ntotal=" << ivpq->ntotal
            << ", is_trained=" << ivpq->is_trained
            << ", metric_type=" << ivpq->metric_type
            << ", nlist=" << ivpq->nlist << ", nprobe="
            << ivpq->nprobe
            // << ", maintain_direct_map=" << ivpq->maintain_direct_map
            << ", by_residual=" << ivpq->by_residual
            << ", code_size=" << ivpq->code_size << ", pq: d=" << ivpq->pq.d
            << ", M=" << ivpq->pq.M << ", nbits=" << ivpq->pq.nbits;

  if (indexed_vec_count_ <= 0) {
    LOG(INFO) << "no vector is indexed, do not need dump";
    return 0;
  }

  /* return rt_invert_index_ptr_->Dump( */
  /*     dir, vec_name, std::min(max_vid, indexed_vec_count_ - 1)); */
  return 0;
}

int GammaIVFPQIndex::Load(const std::vector<std::string> &index_dirs) {
  if (!rt_invert_index_ptr_) {
    return -1;
  }

  string vec_name = raw_vec_->GetName();
  string info_file =
      index_dirs[index_dirs.size() - 1] + "/" + vec_name + ".index.param";
  if (access(info_file.c_str(), F_OK) != 0) {
    LOG(INFO) << info_file << " isn't existed, skip loading";
    return 0;  // it should train again after load
  }

  faiss::IOReader *f = new FileIOReader(info_file.c_str());
  IndexIVFPQ *ivpq = static_cast<IndexIVFPQ *>(this);
  read_ivf_header(ivpq, f, nullptr);  // not legacy
  READ1(ivpq->by_residual);
  READ1(ivpq->code_size);
  read_ProductQuantizer(&ivpq->pq, f);

  // precomputed table not stored. It is cheaper to recompute it
  ivpq->use_precomputed_table = 0;
  if (ivpq->by_residual) ivpq->precompute_table();
  delete f;

  if (!this->is_trained) {
    LOG(ERROR) << "unexpected, gamma index information is loaded, but it "
                  "isn't trained";
    return 0;  // it should train again after load
  }

  /* indexed_vec_count_ = rt_invert_index_ptr_->Load(index_dirs, vec_name); */
  indexed_vec_count_ = 0;

  LOG(INFO) << "load: d=" << ivpq->d << ", ntotal=" << ivpq->ntotal
            << ", is_trained=" << ivpq->is_trained
            << ", metric_type=" << ivpq->metric_type
            << ", nlist=" << ivpq->nlist << ", nprobe="
            << ivpq->nprobe
            // << ", maintain_direct_map=" << ivpq->maintain_direct_map
            << ", by_residual=" << ivpq->by_residual
            << ", code_size=" << ivpq->code_size << ", pq: d=" << ivpq->pq.d
            << ", M=" << ivpq->pq.M << ", nbits=" << ivpq->pq.nbits
            << ", indexed vector count=" << indexed_vec_count_;

  return indexed_vec_count_;
}

}  // namespace tig_gamma
