/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "concurrentqueue/blockingconcurrentqueue.h"
#include "faiss/Index.h"
#include "field_range_index.h"
#include "gamma_gpu_resources.h"
#include "gamma_index.h"
#include "gamma_index_ivfpq.h"
#include "log.h"
#include "raw_vector.h"
#include "utils.h"

namespace tig_gamma {
namespace gamma_gpu {

class GPUItem;

class GammaIVFPQGPUIndex : public GammaIndex {
 public:
  GammaIVFPQGPUIndex(size_t d, size_t nlist, size_t M, size_t nbits_per_idx,
                     const char *docids_bitmap, RawVector<float> *raw_vec,
                     GammaCounters *counters);
  virtual ~GammaIVFPQGPUIndex();

  int Indexing() override;

  int AddRTVecsToIndex() override;

  bool Add(int n, const float *vec) override { return true; };

  int Update(int doc_id, const float *vec) override;

  int Search(const VectorQuery *query, GammaSearchCondition *condition,
             VectorResult &result) override;

  int Delete(int doc_id) override;

  long GetTotalMemBytes() override;

  int Dump(const std::string &dir, int max_vid) override;
  int Load(const std::vector<std::string> &index_dirs) override;

 private:
  int GPUSearch(int n, const float *x, int k, float *distances, long *labels,
                GammaSearchCondition *condition);

  int GPUThread();

  faiss::Index *CreateGPUIndex();

  int CreateSearchThread();

  int indexed_vec_count_;
  size_t nlist_;
  size_t M_;
  size_t nbits_per_idx_;
  int nprobe_;
  int search_idx_;

  moodycamel::BlockingConcurrentQueue<GPUItem *> id_queue_;

  faiss::Index *gpu_index_;
  GammaIVFPQIndex *cpu_index_;

  int tmp_mem_num_;
  std::vector<faiss::gpu::GpuResources *> resources_;
  std::vector<std::thread> gpu_threads_;

  bool b_exited_;

  bool is_indexed_;

  bool use_standard_resource_;

  std::mutex cpu_mutex_;
  std::mutex indexing_mutex_;
#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_count_;
#endif
};

}  // namespace gamma_gpu
}  // namespace tig_gamma
