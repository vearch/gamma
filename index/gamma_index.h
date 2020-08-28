/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef GAMMA_INDEX_H_
#define GAMMA_INDEX_H_

#include <vector>
#include <functional>

#include "gamma_common_data.h"
#include "raw_vector.h"

namespace tig_gamma {

struct VectorResult {
  VectorResult() {
    n = 0;
    topn = 0;
    dists = nullptr;
    docids = nullptr;
    sources = nullptr;
    source_lens = nullptr;
    total.resize(n);
    idx.resize(n);
    idx.assign(n, 0);
    total.assign(n, 0);
  }

  ~VectorResult() {
    if (dists) {
      delete dists;
      dists = nullptr;
    }

    if (docids) {
      delete docids;
      docids = nullptr;
    }

    if (sources) {
      delete sources;
      sources = nullptr;
    }

    if (source_lens) {
      delete source_lens;
      source_lens = nullptr;
    }
  }

  bool init(int a, int b) {
    n = a;
    topn = b;
    dists = new float[n * topn];
    if (!dists) {
      // LOG(ERROR) << "dists in VectorResult malloc error!";
      return false;
    }
    docids = new long[n * topn];
    if (!docids) {
      // LOG(ERROR) << "docids in VectorResult malloc error!";
      return false;
    }

    sources = new char *[n * topn];
    if (!sources) {
      // LOG(ERROR) << "sources in VectorResult malloc error!";
      return false;
    }

    source_lens = new int[n * topn];

    if (!source_lens) {
      // LOG(ERROR) << "source_lens in VectorResult malloc error!";
      return false;
    }

    total.resize(n, 0);
    idx.resize(n, -1);

    return true;
  }

  int seek(const int &req_no, const int &docid, float &score, char *&source,
           int &len) {
    int ret = -1;
    int base_idx = req_no * topn;
    int &start_idx = idx[req_no];
    if (start_idx == -1) return -1;
    for (int i = base_idx + start_idx; i < base_idx + topn; i++) {
      if (docids[i] >= docid) {
        ret = docids[i];
        score = dists[i];
        source = sources[i];
        len = source_lens[i];

        start_idx = i - base_idx;
        break;
      } else {
        continue;
      }
    }
    if (ret == -1) start_idx = -1;
    return ret;
  }

  void sort_by_docid() {
    std::function<int(long *, float *, char **, int *, int, int)> 
      paritition = [&](long *docids, float *dists, char **sources, 
        int *source_lens, int low, int high) {
      long pivot = docids[low];
      float dist = dists[low];
      char *source = sources[low];
      int source_len = source_lens[low];

      while (low < high) {
        while (low < high && docids[high] >= pivot) {
          --high;
        }
        docids[low] = docids[high];
        dists[low] = dists[high];
        sources[low] = sources[high];
        source_lens[low] = source_lens[high];
        while (low < high && docids[low] <= pivot) {
          ++low;
        }
        docids[high] = docids[low];
        dists[high] = dists[low];
        sources[high] = sources[low];
        source_lens[high] = source_lens[low];
      }
      docids[low] = pivot;
      dists[low] = dist;
      sources[low] = source;
      source_lens[low] = source_len;
      return low;
    };

    std::function<void(long *, float *, char **, int *, int, int)> 
        quick_sort_by_docid = [&](long *docids, float *dists, char **sources, int *source_lens, int low, int high) {
      if (low < high) {
        int pivot = paritition(docids, dists, sources, source_lens, low, high);
        quick_sort_by_docid(docids, dists, sources, source_lens, low, pivot - 1);
        quick_sort_by_docid(docids, dists, sources, source_lens, pivot + 1, high);
      }
    };

    for (int i = 0; i < n; ++i) {
      quick_sort_by_docid(docids + i * topn, dists + i * topn, 
        sources + i * topn, source_lens + i * topn, 0, topn - 1);
    }
  }

  int n;
  int topn;
  float *dists;
  long *docids;
  char **sources;
  int *source_lens;
  std::vector<int> total;
  std::vector<int> idx;
};

struct GammaIndex {
  GammaIndex(size_t dimension, const char *docids_bitmap)
      : d_(dimension),
        docids_bitmap_(docids_bitmap),
        raw_vec_(nullptr),
        raw_vec_binary_(nullptr) {}

  void SetRawVectorFloat(RawVector<float> *raw_vec) { raw_vec_ = raw_vec; }
  void SetRawVectorBinary(RawVector<uint8_t> *raw_vec) {
    raw_vec_binary_ = raw_vec;
  }

  virtual ~GammaIndex() {}

  virtual int Indexing() = 0;

  virtual int AddRTVecsToIndex() = 0;
  virtual bool Add(int n, const uint8_t *vec) {
    return true;
  };

  virtual ByteArray *GetBinaryVector(int vec_id) {
    return nullptr;
  };

  virtual bool Add(int n, const float *vec) {
    return true;
  };

  virtual int Update(int doc_id, const float *vec) {
    return 0;
  };

  /** assign the vectors, then call search_preassign */
  virtual int Search(const VectorQuery *query, GammaSearchCondition *condition,
                     VectorResult &result) = 0;

  virtual long GetTotalMemBytes() = 0;

  virtual int Dump(const std::string &dir, int max_vid) = 0;
  virtual int Load(const std::vector<std::string> &index_dirs) = 0;

  virtual int Delete(int docid) = 0;

  int d_;

  const char *docids_bitmap_;
  RawVector<float> *raw_vec_;
  RawVector<uint8_t> *raw_vec_binary_;
};

}  // namespace tig_gamma

#endif
