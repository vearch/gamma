/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_index_binary_ivf.h"

#include "faiss/utils/hamming.h"

namespace tig_gamma {

GammaIndexBinaryIVF::GammaIndexBinaryIVF(faiss::IndexBinary *quantizer,
                                         size_t d, size_t nlist, size_t nprobe,
                                         const char *docids_bitmap,
                                         RawVector<uint8_t> *raw_vec)
    : GammaIndex(d, docids_bitmap), faiss::IndexBinaryIVF(quantizer, d, nlist) {
  raw_vec_binary_ = raw_vec;
  rt_invert_index_ptr_ = new realtime::RTInvertIndex(
      this->nlist, this->code_size, raw_vec->GetMaxVectorSize(),
      raw_vec_binary_->vid_mgr_, docids_bitmap, 10000, 1280000);

  this->nprobe = nprobe;

  if (this->invlists) {
    delete this->invlists;
    this->invlists = nullptr;
  }

  bool ret = rt_invert_index_ptr_->Init();

  if (ret) {
    this->invlists =
        new realtime::RTInvertedLists(rt_invert_index_ptr_, nlist, code_size);
  }
  indexed_vec_count_ = 0;
}

GammaIndexBinaryIVF::~GammaIndexBinaryIVF() {
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

int GammaIndexBinaryIVF::AddRTVecsToIndex() {
  int ret = 0;
  int total_stored_vecs = raw_vec_binary_->GetVectorNum();
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
      size_t count_per_index =
          (i == (index_count - 1) ? total_stored_vecs - start_docid
                                  : MAX_NUM_PER_INDEX);
      ScopeVector<uint8_t> vector_head;
      raw_vec_binary_->GetVectorHeader(indexed_vec_count_,
                                       indexed_vec_count_ + count_per_index,
                                       vector_head);

      uint8_t *add_vec = const_cast<uint8_t *>(vector_head.Get());

      if (!Add(count_per_index, add_vec)) {
        LOG(ERROR) << "add index from docid " << start_docid << " error!";
        ret = -2;
      }
    }
  }
  return ret;
}

bool GammaIndexBinaryIVF::Add(int n, const uint8_t *vec) {
#ifdef PERFORMANCE_TESTING
  double t0 = faiss::getmillisecs();
#endif
  FAISS_THROW_IF_NOT(is_trained);
  assert(invlists);

  std::map<int, std::vector<long>> new_keys;
  std::map<int, std::vector<uint8_t>> new_codes;

  const idx_t *idx;

  std::unique_ptr<idx_t[]> scoped_idx;

  scoped_idx.reset(new idx_t[n]);
  quantizer->assign(n, vec, scoped_idx.get());
  idx = scoped_idx.get();

  size_t n_ignore = 0;
  size_t n_add = 0;
  long vid = indexed_vec_count_;
  for (int i = 0; i < n; i++) {
    long list_no = idx[i];
    assert(list_no < (long)nlist);
    if (list_no < 0) {
      n_ignore++;
      continue;
    }

    // long id = (long)(indexed_vec_count_++);
    const uint8_t *code = vec + i * code_size;

    new_keys[list_no].push_back(vid++);

    size_t ofs = new_codes[list_no].size();
    new_codes[list_no].resize(ofs + code_size);
    memcpy((void *)(new_codes[list_no].data() + ofs), (void *)code, code_size);

    n_add++;
  }

  LOG(WARNING) << "Ignored [" << n_ignore << "]";

  ntotal += n_add;

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
    rt_invert_index_ptr_->PrintBucketSize();
    add_count_ = 0;
  }
#endif
  return true;
}

int GammaIndexBinaryIVF::Indexing() {
  if (this->is_trained) {
    LOG(INFO) << "gamma ivfpq index is already trained, skip indexing";
    return 0;
  }
  int vectors_count = raw_vec_binary_->GetVectorNum();
  if (vectors_count < 8192) {
    LOG(ERROR) << "vector total count [" << vectors_count
               << "] less then 8192, failed!";
    return -1;
  }
  size_t num = vectors_count > 100000 ? 100000 : vectors_count;
  ScopeVector<uint8_t> header;
  raw_vec_binary_->GetVectorHeader(0, num, header);

  uint8_t *train_vec = const_cast<uint8_t *>(header.Get());

  train(num, train_vec);

  LOG(INFO) << "train successed!";
  return 0;
}

int GammaIndexBinaryIVF::Search(const VectorQuery *query,
                                GammaSearchCondition *condition,
                                VectorResult &result) {
  uint8_t *x = reinterpret_cast<uint8_t *>(query->value->value);
  int raw_d = raw_vec_binary_->GetDimension();
  size_t n = query->value->len / (raw_d * sizeof(uint8_t));

  idx_t *idx = reinterpret_cast<idx_t *>(result.docids);

  uint8_t *vec_q = x;

  int32_t dists[n * condition->topn];
  SearchHamming(n, vec_q, condition, dists, idx, result.total.data());
  float *real_score = new float[n * condition->topn];

  for (size_t i = 0; i < n; i++) {
    int pos = 0;

    std::map<int, int> docid2count;
    for (int j = 0; j < condition->topn; j++) {
      result.dists[i * condition->topn + j] = dists[i * condition->topn + j];

      long *docid = result.docids + i * condition->topn + j;
      if (docid[0] == -1) continue;

      int vector_id = (int)docid[0];
      int real_docid = raw_vec_binary_->vid_mgr_->VID2DocID(vector_id);

      if (docid2count.find(real_docid) == docid2count.end()) {
      float score = 1.0 - result.dists[i * condition->topn + j] / float(raw_d *8);

      if (((condition->min_dist >= 0 && score >= condition->min_dist) &&
                  (condition->max_dist >= 0 && score <= condition->max_dist)) ||
                  (condition->min_dist == -1 && condition->max_dist == -1)) {
        int real_pos = i * condition->topn + pos;
        real_score[real_pos] = score;
        result.docids[real_pos] = real_docid;
        int ret = raw_vec_binary_->GetSource(
            vector_id, result.sources[real_pos], result.source_lens[real_pos]);
        if (ret != 0) {
          result.sources[real_pos] = nullptr;
          result.source_lens[real_pos] = 0;
        }
        result.dists[real_pos] = result.dists[i * condition->topn + j];

        pos++;
        docid2count[real_docid] = 1;
        }
      }
    }

    if (pos > 0) {
      result.idx[i] = 0;  // init start id of seeking
    }

    for (; pos < condition->topn; pos++) {
      result.docids[i * condition->topn + pos] = -1;
      result.dists[i * condition->topn + pos] = -1;
      real_score[i * condition->topn + pos] = -1;
    }
  }
  result.dists = real_score;

  return 0;
}

long GammaIndexBinaryIVF::GetTotalMemBytes() {
  if (!rt_invert_index_ptr_) {
    return 0;
  }
  return rt_invert_index_ptr_->GetTotalMemBytes();
}

int GammaIndexBinaryIVF::Delete(int doc_id) {
  std::vector<int> vids;
  raw_vec_binary_->vid_mgr_->DocID2VID(doc_id, vids);
  int ret = rt_invert_index_ptr_->Delete(vids.data(), vids.size());
  return ret;
}

void GammaIndexBinaryIVF::SearchHamming(int n, const uint8_t *x,
                                        GammaSearchCondition *condition,
                                        int32_t *distances, idx_t *labels,
                                        int *total) {
  std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
  std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

  quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

  invlists->prefetch_lists(idx.get(), n * nprobe);

  search_preassigned(n, x, condition, idx.get(), coarse_dis.get(), distances,
                     labels, total, false);
}

void GammaIndexBinaryIVF::search_preassigned(
    int n, const uint8_t *x, GammaSearchCondition *condition, const idx_t *idx,
    const int32_t *coarse_dis, int32_t *distances, idx_t *labels, int *total,
    bool store_pairs, const faiss::IVFSearchParameters *params) {
  search_knn_hamming_heap(n, x, condition, idx, coarse_dis, distances, labels,
                          store_pairs, params);
}

void GammaIndexBinaryIVF::search_knn_hamming_heap(
    size_t n, const uint8_t *x, GammaSearchCondition *condition,
    const idx_t *keys, const int32_t *coarse_dis, int32_t *distances,
    idx_t *labels, bool store_pairs, const faiss::IVFSearchParameters *params) {
  idx_t k = condition->topn;
  long nprobe = params ? params->nprobe : this->nprobe;
  long max_codes = params ? params->max_codes : this->max_codes;

  // almost verbatim copy from IndexIVF::search_preassigned

  using HeapForIP = faiss::CMin<int32_t, idx_t>;
  using HeapForL2 = faiss::CMax<int32_t, idx_t>;

#pragma omp parallel if (n > 1)
  {
    std::unique_ptr<GammaBinaryInvertedListScanner> scanner(
        get_GammaInvertedListScanner(store_pairs));
    scanner->SetVecFilter(docids_bitmap_, raw_vec_binary_);
    scanner->set_search_condition(condition);

#pragma omp for
    for (size_t i = 0; i < n; i++) {
      const uint8_t *xi = x + i * code_size;
      scanner->set_query(xi);

      const idx_t *keysi = keys + i * nprobe;
      int32_t *simi = distances + k * i;
      idx_t *idxi = labels + k * i;

      if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        faiss::heap_heapify<HeapForIP>(k, simi, idxi);
      } else {
        faiss::heap_heapify<HeapForL2>(k, simi, idxi);
      }

      size_t nscan = 0;

      for (long ik = 0; ik < nprobe; ik++) {
        idx_t key = keysi[ik]; /* select the list  */
        if (key < 0) {
          // not enough centroids for multiprobe
          continue;
        }
        FAISS_THROW_IF_NOT_FMT(key < (idx_t)nlist,
                               "Invalid key=%ld  at ik=%ld nlist=%ld\n", key,
                               ik, nlist);

        scanner->set_list(key, coarse_dis[i * nprobe + ik]);

        size_t list_size = invlists->list_size(key);
        faiss::InvertedLists::ScopedCodes scodes(invlists, key);
        std::unique_ptr<faiss::InvertedLists::ScopedIds> sids;
        const faiss::Index::idx_t *ids = nullptr;

        if (!store_pairs) {
          sids.reset(new faiss::InvertedLists::ScopedIds(invlists, key));
          ids = sids->get();
        }

        scanner->scan_codes(list_size, scodes.get(), ids, simi, idxi, k);

        nscan += list_size;
        if (max_codes && nscan >= (size_t)max_codes) break;
      }

      if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        faiss::heap_reorder<HeapForIP>(k, simi, idxi);
      } else {
        faiss::heap_reorder<HeapForL2>(k, simi, idxi);
      }

    }  // parallel for
  }    // parallel
}

template <class HammingComputer, bool store_pairs>
struct GammaIVFBinaryScannerL2 : GammaBinaryInvertedListScanner {
  HammingComputer hc;
  size_t code_size;

  explicit GammaIVFBinaryScannerL2(size_t code_size) : code_size(code_size) {}

  void set_query(const uint8_t *query_vector) override {
    hc.set(query_vector, code_size);
  }

  idx_t list_no;
  void set_list(idx_t list_no, uint8_t /* coarse_dis */) override {
    this->list_no = list_no;
  }

  // uint32_t distance_to_code(const uint8_t *code) const override {
  //   return hc.hamming(code);
  // }

  size_t scan_codes(size_t n, const uint8_t *codes, const idx_t *ids,
                    int32_t *simi, idx_t *idxi, size_t k) const override {
    using C = faiss::CMax<int32_t, idx_t>;

    // set filter func
    std::function<bool(int)> is_filterable;

    if (range_index_ptr_ != nullptr) {
      is_filterable = [this](int doc_id) -> bool {
        return (bitmap::test(docids_bitmap_, doc_id) ||
                (not range_index_ptr_->Has(doc_id)));
      };
    } else {
      is_filterable = [this](int doc_id) -> bool {
        return (bitmap::test(docids_bitmap_, doc_id));
      };
    }

    size_t nup = 0;
    for (size_t j = 0; j < n; j++) {
      idx_t id = store_pairs ? (list_no << 32 | j) : ids[j];
      if (is_filterable(id)) {
        continue;
      }
      uint32_t dis = hc.hamming(codes);
      if (dis < simi[0]) {
        faiss::heap_pop<C>(k, simi, idxi);
        faiss::heap_push<C>(k, simi, idxi, dis, id);
        nup++;
      }
      codes += code_size;
    }
    return nup;
  }
};

template <bool store_pairs>
GammaBinaryInvertedListScanner *select_IVFBinaryScannerL2(size_t code_size) {
  switch (code_size) {
#define HANDLE_CS(cs)                                              \
  case cs:                                                         \
    return new GammaIVFBinaryScannerL2<faiss::HammingComputer##cs, \
                                       store_pairs>(cs);
    HANDLE_CS(4);
    HANDLE_CS(8);
    HANDLE_CS(16);
    HANDLE_CS(20);
    HANDLE_CS(32);
    HANDLE_CS(64);
#undef HANDLE_CS
    default:
      if (code_size % 8 == 0) {
        return new GammaIVFBinaryScannerL2<faiss::HammingComputerM8,
                                           store_pairs>(code_size);
      } else if (code_size % 4 == 0) {
        return new GammaIVFBinaryScannerL2<faiss::HammingComputerM4,
                                           store_pairs>(code_size);
      } else {
        return new GammaIVFBinaryScannerL2<faiss::HammingComputerDefault,
                                           store_pairs>(code_size);
      }
  }
}

GammaBinaryInvertedListScanner *
GammaIndexBinaryIVF::get_GammaInvertedListScanner(bool store_pairs) const {
  if (store_pairs) {
    return select_IVFBinaryScannerL2<true>(code_size);
  } else {
    return select_IVFBinaryScannerL2<false>(code_size);
  }
}

}  // namespace tig_gamma