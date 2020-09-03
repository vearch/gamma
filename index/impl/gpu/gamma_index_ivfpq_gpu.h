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
#include "gamma_index_ivfpq.h"
#include "retrieval_model.h"
#include "log.h"
#include "raw_vector.h"
#include "utils.h"

namespace tig_gamma {
namespace gamma_gpu {

class GPUItem;

class GPURetrievalParameters : public RetrievalParameters {
 public:
  GPURetrievalParameters() : RetrievalParameters() {
    recall_num_ = -1;
    nprobe_ = 20;
  }

  GPURetrievalParameters(size_t recall_num, size_t nprobe,
                         DistanceComputeType type)
      : RetrievalParameters() {
    recall_num_ = recall_num;
    nprobe_ = nprobe;
    distance_compute_type_ = type;
  }
  ~GPURetrievalParameters() {}

  int RecallNum() { return recall_num_; }
  void SetRecallNum(int recall_num) { recall_num_ = recall_num; }
  int Nprobe() { return nprobe_; }
  void SetNprobe(int nprobe) { nprobe_ = nprobe; }

 protected:
  int recall_num_;
  int nprobe_;
};

class GammaIVFPQGPUIndex : public RetrievalModel {
 public:
  GammaIVFPQGPUIndex(VectorReader *vec, const std::string &model_parameters);

  GammaIVFPQGPUIndex();

  virtual ~GammaIVFPQGPUIndex();

  int Init(const std::string &model_parameters);

  RetrievalParameters *Parse(const std::string &parameters);

  int Indexing() override;

  int AddRTVecsToIndex();

  bool Add(int n, const uint8_t *vec) override;

  int Update(const std::vector<int64_t> &ids,
             const std::vector<const uint8_t *> &vecs) override;

  int Search(RetrievalContext *retrieval_context, int n, const uint8_t *x,
             int k, float *distances, int64_t *labels);

  int Delete(const std::vector<int64_t> &ids) override;

  long GetTotalMemBytes() override;

  int Dump(const std::string &dir) override;
  int Load(const std::string &index_dir) override;

 private:
  int GPUThread();

  faiss::Index *CreateGPUIndex();

  int CreateSearchThread();

  int indexed_vec_count_;
  size_t nlist_;
  size_t M_;
  size_t nbits_per_idx_;
  int nprobe_;

  moodycamel::BlockingConcurrentQueue<GPUItem *> id_queue_;

  faiss::Index *gpu_index_;
  GammaIVFPQIndex *cpu_index_;

  int tmp_mem_num_;
  std::vector<faiss::gpu::GpuResources *> resources_;
  std::vector<std::thread> gpu_threads_;

  bool b_exited_;

  bool is_trained_;

  bool use_standard_resource_;
  int d_;
  std::mutex cpu_mutex_;
  std::mutex indexing_mutex_;
#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_count_;
#endif
};

}  // namespace gamma_gpu
}  // namespace tig_gamma
