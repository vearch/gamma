// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_SCANN_OPS_CC_SCANN_H_
#define SCANN_SCANN_OPS_CC_SCANN_H_

#include <cstdint>
#include <limits>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/threads.h"

namespace research_scann {

class ScannInterface {
 public:
  ScannInterface(std::string str_config) { str_config_ = str_config; }
  ScannInterface() {}

  Status Initialize(ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points,
                    const std::string& artifacts_dir);
  Status Initialize(ScannConfig config, SingleMachineFactoryOptions opts,
                    ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points);
  Status Initialize(ConstSpan<float> dataset, DatapointIndex n_points,
                    const std::string& config, int training_threads);
  Status Initialize(
      shared_ptr<DenseDataset<float>> dataset,
      SingleMachineFactoryOptions &opts);
//      SingleMachineFactoryOptions opts = SingleMachineFactoryOptions());


  Status Search(const DatapointPtr<float> query, NNResultsVector* res,
                int final_nn, int pre_reorder_nn, int leaves) const;
  Status SearchBatched(const DenseDataset<float>& queries,
                       MutableSpan<NNResultsVector> res, int final_nn,
                       int pre_reorder_nn, int leaves) const;
  Status SearchBatchedParallel(const DenseDataset<float>& queries,
                               MutableSpan<NNResultsVector> res, int final_nn,
                               int pre_reorder_nn, int leaves) const;

  int AddBatched(std::vector<float> &dataset, uint32_t doc_num);

  int DumpIndex(std::string path);

  int BuildIndex();

  Status Serialize(std::string path);
  StatusOr<SingleMachineFactoryOptions> ExtractOptions();

  template <typename T_idx>
  void ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                       float* distances);
  template <typename T_idx>
  void ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res, T_idx* indices,
                              float* distances);

  bool needs_dataset() const { return scann_->needs_dataset(); }
  const Dataset* dataset() const { return scann_->dataset(); }

  size_t n_points() const { return n_points_; }
  DimensionIndex dimensionality() const { return dimensionality_; }
  const ScannConfig* config() const { return &config_; }

  std::string str_config() { return str_config_; }

 private:
  size_t n_points_;
  shared_ptr<DenseDataset<float>> dataset_;
  DimensionIndex dimensionality_;
  std::unique_ptr<SingleMachineSearcherBase<float>> scann_;
  ScannConfig config_;
  SingleMachineFactoryOptions opts_;
  std::string str_config_;


  float result_multiplier_;

  size_t min_batch_size_;
};

template <typename T_idx>
void ScannInterface::ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                                     float* distances) {
  for (const auto& p : res) {
    *(indices++) = static_cast<T_idx>(p.first);
    *(distances++) = result_multiplier_ * p.second;
  }
}

template <typename T_idx>
void ScannInterface::ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res,
                                            T_idx* indices, float* distances) {
  for (const auto& result_vec : res) {
    for (const auto& pair : result_vec) {
      *(indices++) = static_cast<T_idx>(pair.first);
      *(distances++) = result_multiplier_ * pair.second;
    }
  }
}

}  // namespace research_scann

#endif
