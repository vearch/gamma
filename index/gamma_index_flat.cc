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
#include "gamma_index_flat.h"

namespace tig_gamma {

GammaFLATIndex::GammaFLATIndex(size_t d, const char *docids_bitmap,
                               RawVector<float> *raw_vec)
    : GammaIndex(d, docids_bitmap) {
  SetRawVectorFloat(raw_vec);
}

GammaFLATIndex::~GammaFLATIndex() {}

int GammaFLATIndex::Indexing() { return 0; }

int GammaFLATIndex::AddRTVecsToIndex() { return 0; }

bool GammaFLATIndex::Add(int n, const float *vec) { return 0; }

int GammaFLATIndex::Search(const VectorQuery *query,
                           GammaSearchCondition *condition,
                           VectorResult &result) {

  float *x = reinterpret_cast<float *>(query->value->value);
  int raw_d = raw_vec_->GetDimension();
  size_t n = query->value->len / (raw_d * sizeof(float));

  idx_t *idx = reinterpret_cast<idx_t *>(result.docids);

  SearchDirectly(n, x, condition, result.dists, idx, result.total.data());

  for (size_t i = 0; i < n; i++) {
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

void GammaFLATIndex::SearchDirectly(int n, const float *x,
                                     GammaSearchCondition *condition,
                                     float *distances, idx_t *labels,
                                     int *total) {

  int num_vectors = raw_vec_->GetVectorNum();
  ScopeVector<float> scope_vec;
  raw_vec_->GetVectorHeader(0, 0 + num_vectors, scope_vec);
  const float *vectors = scope_vec.Get();

  long k = condition->topn;  // topK

  int d = raw_vec_->GetDimension();
  if (condition->metric_type == InnerProduct) {
    metric_type_ = faiss::METRIC_INNER_PRODUCT;
  } else {
    metric_type_ = faiss::METRIC_L2;
  }
  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  {
    // we must obtain the num of threads in *THE* parallel area.
    int num_threads = omp_get_max_threads();

    /*****************************************************
     * Depending on parallel_mode, there are two possible ways
     * to organize the search. Here we define local functions
     * that are in common between the two
     ******************************************************/

    auto init_result = [&](int k, float *simi, idx_t *idxi) {
      if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
        faiss::heap_heapify<HeapForIP>(k, simi, idxi);
      } else {
        faiss::heap_heapify<HeapForL2>(k, simi, idxi);
      }
    };

    auto reorder_result = [&](int k, float *simi, idx_t *idxi) {
      if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
        faiss::heap_reorder<HeapForIP>(k, simi, idxi);
      } else {
        faiss::heap_reorder<HeapForL2>(k, simi, idxi);
      }
    };

    auto sort_by_docid = [&](int k, float *simi, idx_t *idxi) {
      std::vector<std::pair<long, float>> id_sim_pairs;
      for (int i = 0; i < k; i++) {
        id_sim_pairs.emplace_back(std::make_pair(idxi[i], simi[i]));
      }
      std::sort(id_sim_pairs.begin(), id_sim_pairs.end());
      for (int i = 0; i < k; i++) {
        idxi[i] = id_sim_pairs[i].first;
        simi[i] = id_sim_pairs[i].second;
      }
    };

    auto search_impl = [&](const float *xi, const float *y, int ny, int offset,
                           float *simi, idx_t *idxi, int k) -> int {
      int total = 0;
      auto *nr = condition->range_query_result;
      bool ck_dis = (condition->min_dist >= 0 && condition->max_dist >= 0);

      if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
        for (int i = 0; i < ny; i++) {
          int vid = offset + i;
          auto docid = raw_vec_->vid_mgr_->VID2DocID(vid);

          if (bitmap::test(docids_bitmap_, docid) ||
              (nr && not nr->Has(docid))) {
            continue;
          }

          const float *yi = y + i * d;
          float dis = faiss::fvec_inner_product(xi, yi, d);

          if (ck_dis &&
              (dis < condition->min_dist || dis > condition->max_dist)) {
            continue;
          }

          if (HeapForIP::cmp(simi[0], dis)) {
            faiss::heap_pop<HeapForIP>(k, simi, idxi);
            faiss::heap_push<HeapForIP>(k, simi, idxi, dis, vid);
          }

          total++;
        }
      } else {
        for (int i = 0; i < ny; i++) {
          int vid = offset + i;
          auto docid = raw_vec_->vid_mgr_->VID2DocID(vid);

          if (bitmap::test(docids_bitmap_, docid) ||
              (nr && not nr->Has(docid))) {
            continue;
          }

          const float *yi = y + i * d;
          float dis = faiss::fvec_L2sqr(xi, yi, d);

          if (ck_dis &&
              (dis < condition->min_dist || dis > condition->max_dist)) {
            continue;
          }

          if (HeapForL2::cmp(simi[0], dis)) {
            faiss::heap_pop<HeapForL2>(k, simi, idxi);
            faiss::heap_push<HeapForL2>(k, simi, idxi, dis, vid);
          }

          total++;
        }
      }

      return total;
    };

    if (condition->parallel_mode == 0) {  // parallelize over queries
#pragma omp for
      for (int i = 0; i < n; i++) {
        const float *xi = x + i * d;

        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;

        init_result(k, simi, idxi);

        total[i] += search_impl(xi, vectors, num_vectors, 0, simi, idxi, k);

        if (condition->sort_by_docid) {
          sort_by_docid(k, simi, idxi);
        } else {  // sort by dist
          reorder_result(k, simi, idxi);
        }
      }
    } else {  // parallelize over vectors

      size_t num_vectors_per_thread = num_vectors / num_threads;

      for (int i = 0; i < n; i++) {
        const float *xi = x + i * d;

        // merge thread-local results
        float *simi = distances + i * k;
        idx_t *idxi = labels + i * k;
        init_result(k, simi, idxi);

        size_t ndis = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : ndis)
        for (int ik = 0; ik < num_threads; ik++) {
          std::vector<idx_t> local_idx(k);
          std::vector<float> local_dis(k);
          init_result(k, local_dis.data(), local_idx.data());

          const float *y = vectors + ik * num_vectors_per_thread * d;
          size_t ny = num_vectors_per_thread;

          if (ik == num_threads - 1) {
            ny += num_vectors % num_threads;  // the rest
          }

          int offset = ik * num_vectors_per_thread;

          ndis += search_impl(xi, y, ny, offset, local_dis.data(),
                              local_idx.data(), k);

#pragma omp critical
          {
            if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
              faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis.data(),
                                          local_idx.data(), k);
            } else {
              faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis.data(),
                                          local_idx.data(), k);
            }
          }
        }

        total[i] += ndis;

        if (condition->sort_by_docid) {
          sort_by_docid(k, simi, idxi);
        } else {
          reorder_result(k, simi, idxi);
        }
      }
    }
  }  // parallel

#ifdef PERFORMANCE_TESTING
  std::string compute_msg = "compute ";
  compute_msg += std::to_string(n);
  condition->Perf(compute_msg);
#endif // PERFORMANCE_TESTING
}

long GammaFLATIndex::GetTotalMemBytes() { return 0; }

int GammaFLATIndex::Update(int doc_id, const float *vec) { return 0; }

int GammaFLATIndex::Dump(const std::string &dir, int max_vid) { return 0; }

int GammaFLATIndex::Load(const std::vector<std::string> &index_dirs) {
  return 0;
}

}  // namespace tig_gamma
