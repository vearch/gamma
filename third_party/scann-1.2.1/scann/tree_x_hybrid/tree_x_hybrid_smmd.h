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



#ifndef SCANN_TREE_X_HYBRID_TREE_X_HYBRID_SMMD_H_
#define SCANN_TREE_X_HYBRID_TREE_X_HYBRID_SMMD_H_

#include <cstdint>
#include <functional>

#include "absl/synchronization/mutex.h"
#include "gtest/gtest_prod.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename U>
class DisjointRestrictTokenSearcher;

template <typename T>
class TreeXHybridSMMD : public SingleMachineSearcherBase<T> {
 public:
  TreeXHybridSMMD(shared_ptr<const TypedDataset<T>> dataset,
                  shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
                  int32_t default_pre_reordering_num_neighbors,
                  float default_pre_reordering_epsilon);

  ~TreeXHybridSMMD() {
    int n_tokens = leaf_searcher_locks_.size();
    for (int i = 0; i < n_tokens; ++i) {
      pthread_rwlock_destroy(&leaf_searcher_locks_[i]);
      if (datapoints_by_token_.size() > i && datapoints_by_token_[i]) {
        delete datapoints_by_token_[i];
        datapoints_by_token_[i];
      }
    }
    pthread_rwlock_destroy(&dataset_rwlock_);
  }

  using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

  DatapointIndex optimal_batch_size() const final;

  Status BuildLeafSearchers(
      const Partitioner<T>& database_tokenizer,
      std::function<StatusOrSearcher(
          shared_ptr<TypedDataset<T>> dataset_partition,
          shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
          int32_t token)>
          leaf_searcher_builder) {
    return BuildLeafSearchers(database_tokenizer, leaf_searcher_builder,
                              nullptr);
  }

  Status BuildLeafSearchers(
      const Partitioner<T>& database_tokenizer,
      std::function<StatusOrSearcher(
          shared_ptr<TypedDataset<T>> dataset_partition,
          shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
          int32_t token)>
          leaf_searcher_builder,
      shared_ptr<ThreadPool> thread_pool);

  Status BuildLeafSearchers(
      vector<std::vector<DatapointIndex>> datapoints_by_token,
      std::function<StatusOrSearcher(
          shared_ptr<TypedDataset<T>> dataset_partition,
          shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
          int32_t token)>
          leaf_searcher_builder);

  Status BuildPretrainedScalarQuantizationLeafSearchers(
      vector<std::vector<DatapointIndex>> datapoints_by_token,
      vector<DenseDataset<int8_t>> partitioned_datasets,
      vector<vector<float>> partitioned_squared_l2_norms,
      std::function<
          StatusOrSearcher(DenseDataset<int8_t> scalar_quantized_partition,
                           vector<float> squared_l2_norms)>
          leaf_searcher_builder);

  void set_query_tokenizer(shared_ptr<const Partitioner<T>> query_tokenizer) {
    query_tokenizer_ = query_tokenizer;
  }

  shared_ptr<const Partitioner<T>> query_tokenizer() {
    return query_tokenizer_;
  }

  void set_database_tokenizer(
      shared_ptr<const Partitioner<T>> database_tokenizer) {
    database_tokenizer_ = database_tokenizer;
  }

  shared_ptr<const Partitioner<T>> database_tokenizer() {
    return database_tokenizer_;
  }

  void set_leaf_searcher_optional_parameter_creator(
      shared_ptr<const LeafSearcherOptionalParameterCreator<T>> x);

  ConstSpan<std::vector<DatapointIndex> *> datapoints_by_token() const {
    return ConstSpan<std::vector<DatapointIndex> *>(datapoints_by_token_);
  }

  ConstSpan<unique_ptr<SingleMachineSearcherBase<T>>> leaf_searchers() {
    return ConstSpan<unique_ptr<SingleMachineSearcherBase<T>>>(
        leaf_searchers_.data(), leaf_searchers_.size());
  }

  bool supports_crowding() const final { return true; }

  Status PreprocessQueryIntoParamsUnlocked(
      const DatapointPtr<T>& query,
      SearchParameters& search_params) const final;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

  pthread_rwlock_t *GetDatasetRwlock() override { return &dataset_rwlock_; }

  int KmeanTokenizeDatabase(
      shared_ptr<DenseDataset<T>> add_dataset,
      vector<std::vector<DatapointIndex>> &datapoints_by_token,
      ThreadPool* pool) override;

  int AddDataset(shared_ptr<DenseDataset<T>> raw_dataset,
      shared_ptr<DenseDataset<T>> add_dataset) override;

  int AddDatapointsForOneLeaf(std::vector<DatapointIndex> &datapoints_by_token,
                                      uint32_t bucket_id) override;

  int AddPackedDataset(
      const shared_ptr<DenseDataset<T>> add_dataset,
      vector<std::vector<DatapointIndex>> &datapoints_by_token,
      shared_ptr<ThreadPool> pool) override;

  int DumpIndex(std::string path) {
    LOG(INFO) << "SingleMachineSearcherBase::DumpIndex";
    return 0; 
  }

  void SetIndexer(shared_ptr<const asymmetric_hashing2::Indexer<T>> indexer) {
    indexer_ = indexer;
  }

  void SetnoiseShapingThreshold(double noise_shaping_threshold) {
    noise_shaping_threshold_  = noise_shaping_threshold;
  }

 protected:
  bool impl_needs_dataset() const final { return leaf_searchers_.empty(); }

  bool impl_needs_hashed_dataset() const final {
    return leaf_searchers_.empty();
  }

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute) final;
  void DisableCrowdingImpl() final;

  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

 private:
  class CentersToSearch
      : public SearchParameters::UnlockedQueryPreprocessingResults {
   public:
    explicit CentersToSearch(vector<int32_t> centers_to_search)
        : centers_to_search_(std::move(centers_to_search)) {}

    ConstSpan<int32_t> centers_to_search() const { return centers_to_search_; }

   private:
    vector<int32_t> centers_to_search_;
  };

  Status CheckReadyToQuery(const SearchParameters& params) const;

  Status ValidateTokenList(ConstSpan<int32_t> token_list, bool check_oob) const;

  template <typename TopN>
  Status FindNeighborsPreTokenizedImpl(const DatapointPtr<T>& query,
                                       const SearchParameters& params,
                                       ConstSpan<int32_t> query_tokens,
                                       TopN top_n,
                                       NNResultsVector* results) const;

  Status FindNeighborsPreTokenizedBatchedGenericImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      ConstSpan<ConstSpan<int32_t>> query_tokens,
      MutableSpan<NNResultsVector> results) const;

  Status FindNeighborsPreTokenizedBatchedOptimizedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      ConstSpan<ConstSpan<int32_t>> query_tokens,
      MutableSpan<NNResultsVector> results) const;

  StatusOr<pair<int32_t, DatapointPtr<T>>> TokenizeAndMaybeResidualize(
      const DatapointPtr<T>& dptr, Datapoint<T>*);

  StatusOr<vector<pair<int32_t, DatapointPtr<T>>>> TokenizeAndMaybeResidualize(
      const TypedDataset<T>& dps, MutableSpan<Datapoint<T>*>);

  StatusOr<shared_ptr<const SearcherSpecificOptionalParameters>>
  CreateLeafOptionalParameters(const DatapointPtr<T>& query,
                               const SearchParameters& top_level_params) const;

  vector<unique_ptr<SingleMachineSearcherBase<T>>> leaf_searchers_;

  shared_ptr<const Partitioner<T>> query_tokenizer_;
  shared_ptr<const Partitioner<T>> database_tokenizer_;

  vector<std::vector<DatapointIndex> *> datapoints_by_token_;

  bool is_train_ = false;
  shared_ptr<const asymmetric_hashing2::Indexer<T>> indexer_;
  double noise_shaping_threshold_ = NAN;
  vector<pthread_rwlock_t> leaf_searcher_locks_;
  pthread_rwlock_t dataset_rwlock_;
  uint8_t dataset_extend_time_ = 0;

  shared_ptr<const LeafSearcherOptionalParameterCreator<T>>
      leaf_searcher_optional_parameter_creator_ = nullptr;

  bool disjoint_leaf_partitions_ = true;

  DatapointIndex num_datapoints_ = 0;

  template <typename U>
  friend class DisjointRestrictTokenSearcher;
};

#define SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_CROWDING(extern_keyword, data_type)
#define SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword,      \
                                                      data_type)           \
  extern_keyword template class TreeXHybridSMMD<data_type>;                \
  extern_keyword template Status TreeXHybridSMMD<data_type>::              \
      FindNeighborsPreTokenizedImpl<TopNeighbors<float>>(                  \
          const DatapointPtr<data_type>& query,                            \
          const SearchParameters& params, ConstSpan<int32_t> query_tokens, \
          TopNeighbors<float> top_n, NNResultsVector* results) const;      \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_CROWDING(extern_keyword, data_type)
#define SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD(extern_keyword)               \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, int8_t);   \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, uint8_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, int16_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, uint16_t); \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, int32_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, uint32_t); \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, int64_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, uint64_t); \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, float);    \
  SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD_FOR_TYPE(extern_keyword, double);

SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD(extern);

}  // namespace research_scann

#endif
