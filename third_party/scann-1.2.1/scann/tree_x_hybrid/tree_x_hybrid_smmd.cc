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



#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/serialization.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/tree_x_hybrid/internal/batching.h"
#include "scann/tree_x_hybrid/internal/utils.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/hash_leaf_helpers.h"

namespace research_scann {

template <typename T>
TreeXHybridSMMD<T>::TreeXHybridSMMD(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase<T>(dataset, hashed_dataset,
                                   default_pre_reordering_num_neighbors,
                                   default_pre_reordering_epsilon) {}
template <typename T>
DatapointIndex TreeXHybridSMMD<T>::optimal_batch_size() const {
  auto kmeans_partitioner =
      dynamic_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
  if (!kmeans_partitioner) return 1;
  return (kmeans_partitioner->SupportsLowLevelQueryBatching()) ? 256 : 1;
}

namespace {

template <typename T>
TypedDataset<T>* PartitionDataset(const TypedDataset<T>& original,
                                  ConstSpan<DatapointIndex> subset) {
  TypedDataset<T>* result =
      (original.IsSparse())
          ? absl::implicit_cast<TypedDataset<T>*>(new SparseDataset<T>)
          : absl::implicit_cast<TypedDataset<T>*>(new DenseDataset<T>);
  result->set_packing_strategy(original.packing_strategy());
  result->set_dimensionality(original.dimensionality());
  result->Reserve(subset.size());
  for (const DatapointIndex i : subset) {
    result->AppendOrDie(original[i], "");
  }

  result->set_normalization_tag(original.normalization());
  return result;
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::BuildLeafSearchers(
    const Partitioner<T>& database_tokenizer,
    std::function<StatusOrSearcher(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder,
    shared_ptr<ThreadPool> thread_pool) {
  if (!leaf_searchers_.empty()) {
    return FailedPreconditionError(
        "BuildLeafSearchers must not be called more than once per instance.");
  }

  VLOG(1) << "Tokenizing database...";
  const absl::Time tokenization_start = absl::Now();
  TF_ASSIGN_OR_RETURN(
      auto datapoints_by_token,
      database_tokenizer.TokenizeDatabase(*this->dataset(), thread_pool.get()));
  VLOG(1) << "Done tokenizing database in " << absl::Now() - tokenization_start
          << ".";
  return BuildLeafSearchers(std::move(datapoints_by_token),
                            leaf_searcher_builder);
}

template <typename T>
void TreeXHybridSMMD<T>::set_leaf_searcher_optional_parameter_creator(
    shared_ptr<const LeafSearcherOptionalParameterCreator<T>> x) {
  leaf_searcher_optional_parameter_creator_ = std::move(x);
}

namespace {

Status ValidateDatapointsByToken(
    const vector<std::vector<DatapointIndex>>& datapoints_by_token,
    DatapointIndex num_datapoints, bool* is_disjoint) {
  DCHECK(is_disjoint);
  *is_disjoint = true;

  vector<bool> global_bitmap(num_datapoints, false);

  for (const std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
    DCHECK(std::is_sorted(dp_list.begin(), dp_list.end()));
    auto duplicate_it = std::adjacent_find(dp_list.begin(), dp_list.end());
    if (duplicate_it != dp_list.end()) {
      return InvalidArgumentError(
          absl::StrCat("Duplicate datapoint index within a partition of "
                       "datapoints_by_token:  ",
                       *duplicate_it, "."));
    }

    for (DatapointIndex dp_index : dp_list) {
      if (dp_index >= num_datapoints) {
        return OutOfRangeError(absl::StrCat(
            "Datapoint index in datapoints_by_token is >= number of "
            "datapoints in database (",
            dp_index, " vs. ", num_datapoints, ")."));
      }

      if (global_bitmap[dp_index]) {
        *is_disjoint = false;
      } else {
        global_bitmap[dp_index] = true;
      }
    }
  }

  const DatapointIndex num_missing =
      std::count(global_bitmap.begin(), global_bitmap.end(), false);
  if (num_missing > 0) {
    auto false_it =
        std::find(global_bitmap.begin(), global_bitmap.end(), false);
    const size_t first_missing = false_it - global_bitmap.begin();
    return InvalidArgumentError(absl::StrCat(
        "Found ", num_missing,
        " datapoint(s) "
        "that are not represented in any partition.  First missing "
        "datapoint index = ",
        first_missing, "."));
  }

  return OkStatus();
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::BuildLeafSearchers(
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    std::function<StatusOrSearcher(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder) {
  if (0 != pthread_rwlock_init(&dataset_rwlock_, nullptr)) {
    LOG(ERROR) << "pthread_rwlock_init dataset_rwlock_ error";
  }
  leaf_searcher_locks_.resize(datapoints_by_token.size());
  for (int k = 0; k < leaf_searcher_locks_.size(); ++k) {
    if (0 != pthread_rwlock_init(&leaf_searcher_locks_[k], nullptr)) {
      LOG(ERROR) << "pthread_rwlock_init leaf_searcher_locks_ error, lock_id:" << k;
    }
  }

  for (std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
    std::sort(dp_list.begin(), dp_list.end());
    if (!dp_list.empty()) {
      num_datapoints_ = std::max(num_datapoints_, dp_list.back() + 1);
    }
  }

  auto dataset_size_or_error = this->DatasetSize();
  SCANN_RETURN_IF_ERROR(dataset_size_or_error.status());
  const DatapointIndex dataset_size = dataset_size_or_error.ValueOrDie();
  SCANN_RETURN_IF_ERROR(ValidateDatapointsByToken(
      datapoints_by_token, dataset_size, &disjoint_leaf_partitions_));
  DatapointIndex total_leaf_partition_size = 0;
  for (const auto& token_datapoints : datapoints_by_token) {
    total_leaf_partition_size += token_datapoints.size();
  }

  VLOG(1) << "Original dataset size = " << dataset_size
          << ", sum of leaf partition sizes = " << total_leaf_partition_size;

  const TypedDataset<T>* dataset = this->dataset();
  const DenseDataset<uint8_t>* hashed_dataset = this->hashed_dataset();
  const DatapointIndex n_tokens = datapoints_by_token.size();
  leaf_searchers_.resize(n_tokens);
  for (int32_t token = 0; token < n_tokens; ++token) {
    const absl::Time token_start = absl::Now();
    if (!hashed_dataset) {
      shared_ptr<TypedDataset<T>> dataset_partition(
          PartitionDataset(*dataset, datapoints_by_token[token]));
      TF_ASSIGN_OR_RETURN(
          unique_ptr<SingleMachineSearcherBase<T>> leaf_searcher,
          leaf_searcher_builder(dataset_partition, nullptr, token));

      if (!leaf_searcher->needs_dataset()) {
        leaf_searcher->ReleaseDatasetAndDocids();
      }

      leaf_searchers_[token] = std::move(leaf_searcher);
    } else {
      shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition(
          down_cast<DenseDataset<uint8_t>*>(
              PartitionDataset(*hashed_dataset, datapoints_by_token[token])));
      TF_ASSIGN_OR_RETURN(
          unique_ptr<SingleMachineSearcherBase<T>> leaf_searcher,
          leaf_searcher_builder(nullptr, hashed_dataset_partition, token));
      if (!leaf_searcher->needs_hashed_dataset()) {
        leaf_searcher->ReleaseHashedDataset();
      }
      leaf_searchers_[token] = std::move(leaf_searcher);
    }
    leaf_searchers_[token]->SetWrlock(&leaf_searcher_locks_[token]);

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
  }
  datapoints_by_token_.resize(n_tokens);
  for (int i = 0; i < n_tokens; ++i) {
    datapoints_by_token_[i] = new std::vector<DatapointIndex>(std::move(datapoints_by_token[i]));
  }
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute());
  }
  is_train_ = true;
  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildPretrainedScalarQuantizationLeafSearchers(
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    vector<DenseDataset<int8_t>> partitioned_datasets,
    vector<vector<float>> partitioned_squared_l2_norms,
    std::function<
        StatusOrSearcher(DenseDataset<int8_t> scalar_quantized_partition,
                         vector<float> squared_l2_norms)>
        leaf_searcher_builder) {
  for (auto& dp_list : datapoints_by_token) {
    std::sort(dp_list.begin(), dp_list.end());
    if (!dp_list.empty()) {
      num_datapoints_ = std::max(num_datapoints_, dp_list.back() + 1);
    }
  }

  const auto n_tokens = datapoints_by_token.size();
  leaf_searchers_.resize(n_tokens);
  for (int32_t token = 0; token < n_tokens; ++token) {
    const auto token_start = absl::Now();

    vector<float> squared_l2_norms;
    if (!partitioned_squared_l2_norms.empty())
      squared_l2_norms = std::move(partitioned_squared_l2_norms[token]);
    TF_ASSIGN_OR_RETURN(
        auto leaf_searcher,
        leaf_searcher_builder(std::move(partitioned_datasets[token]),
                              std::move(squared_l2_norms)));
    leaf_searchers_[token] = std::move(leaf_searcher);
    leaf_searchers_[token]->SetWrlock(&leaf_searcher_locks_[token]);

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
  }

  datapoints_by_token_.resize(n_tokens);
  for (int i = 0; i < n_tokens; ++i) {
    datapoints_by_token_[i] = new std::vector<DatapointIndex>(std::move(datapoints_by_token[i]));
  }
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute());
  }
  is_train_ = true;

  return OkStatus();
}

namespace {

void RemapToGlobalDatapointIndices(
    MutableSpan<pair<DatapointIndex, float>> partition_leaf_result,
    ConstSpan<DatapointIndex> local_to_global_datapoint_indices) {
  for (auto& r : partition_leaf_result) {
    r.first = local_to_global_datapoint_indices[r.first];
  }
}

template <typename TopN>
void MergeLeafResultsWithDuplicates(ConstSpan<NNResultsVector> to_merge,
                                    TopN top_n, NNResultsVector* result) {
  DCHECK(result);
  DCHECK(top_n.empty());

  absl::node_hash_map<DatapointIndex, pair<float, int32_t>> duplicates_merged;
  for (const auto& v : to_merge) {
    for (const auto& neighbor : v) {
      pair<float, int32_t>& val = duplicates_merged[neighbor.first];
      ++val.second;
      val.first += neighbor.second;
    }
  }
  for (const auto& elem : duplicates_merged) {
    const float averaged_dist = elem.second.first / elem.second.second;
    top_n.push(std::make_pair(elem.first, averaged_dist));
  }
  *result = top_n.TakeUnsorted();
}

inline bool PreTokenizationEnabled(
    const shared_ptr<const TreeXOptionalParameters>& params) {
  return params && params->pre_tokenization_enabled();
}

inline bool PreTokenizationEnabled(const SearchParameters& params) {
  return PreTokenizationEnabled(
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>());
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (leaf_searchers_.empty()) return OkStatus();
  for (size_t token = 0; token < leaf_searchers_.size(); ++token) {
    ConstSpan<DatapointIndex> cur_leaf_datapoints = *datapoints_by_token_[token];
    vector<int64_t> leaf_datapoint_index_to_crowding_attribute(
        cur_leaf_datapoints.size());
    for (size_t i = 0; i < cur_leaf_datapoints.size(); ++i) {
      leaf_datapoint_index_to_crowding_attribute[i] =
          datapoint_index_to_crowding_attribute[cur_leaf_datapoints[i]];
    }
    Status status = leaf_searchers_[token]->EnableCrowding(
        std::move(leaf_datapoint_index_to_crowding_attribute));
    if (!status.ok()) {
      for (size_t i = 0; i <= token; ++i) {
        leaf_searchers_[i]->DisableCrowding();
      }
    }
  }
  return OkStatus();
}

template <typename T>
void TreeXHybridSMMD<T>::DisableCrowdingImpl() {
  for (auto& ls : leaf_searchers_) {
    ls->DisableCrowding();
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsImpl(const DatapointPtr<T>& query,
                                             const SearchParameters& params,
                                             NNResultsVector* result) const {
  SCANN_RETURN_IF_ERROR(CheckReadyToQuery(params));
  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> query_tokens_storage;
  ConstSpan<int32_t> query_tokens;
  if (PreTokenizationEnabled(tree_x_params)) {
    query_tokens = tree_x_params->leaf_tokens_to_search();
  } else if (params.unlocked_query_preprocessing_results<CentersToSearch>()) {
    query_tokens =
        params.unlocked_query_preprocessing_results<CentersToSearch>()
            ->centers_to_search();
  } else {
    bool override = false;

    if (tree_x_params) {
      const auto num_partitions_to_search_override =
          tree_x_params->num_partitions_to_search_override();
      if (num_partitions_to_search_override > 0) {
        const auto* kmeans_tokenizer =
            down_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
        if (!kmeans_tokenizer)
          return InvalidArgumentError(
              "num_partitions_to_search_override is > 0, but the tokenizer is "
              "not"
              " a KMeansTreePartitioner.");

        SCANN_RETURN_IF_ERROR(
            kmeans_tokenizer->TokensForDatapointWithSpillingAndOverride(
                query, num_partitions_to_search_override,
                &query_tokens_storage));
        override = true;
      }
    }

    if (!override) {
      SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
          query, &query_tokens_storage));
    }

    query_tokens = query_tokens_storage;
  }

  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    return FindNeighborsPreTokenizedImpl(
        query, params, query_tokens,
        TopNeighbors<float>(params.pre_reordering_num_neighbors()), result);
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if (params.empty()) {
    DCHECK_EQ(queries.size(), 0);
    DCHECK_EQ(results.size(), 0);
    return OkStatus();
  }

  SCANN_RETURN_IF_ERROR(CheckReadyToQuery(params[0]));

  const bool pre_tokenized = PreTokenizationEnabled(params[0]);
  for (const SearchParameters& p : params) {
    if (PreTokenizationEnabled(p) != pre_tokenized) {
      return InvalidArgumentError(
          "For tree-X hybrid FindNeighborsBatched, either all queries in a "
          "given batch must use pre-tokenization or no queries in this batch "
          "must use pre-tokenization.");
    }
  }

  const bool has_unlocked_preprocessing =
      params[0].unlocked_query_preprocessing_results<CentersToSearch>();
  if (!pre_tokenized) {
    for (const SearchParameters& p : params) {
      if (static_cast<bool>(
              p.unlocked_query_preprocessing_results<CentersToSearch>()) !=
          has_unlocked_preprocessing) {
        return InvalidArgumentError(
            "For tree-X hybrid FindNeighborsBatched, either all queries in a "
            "given batch must use unlocked preprocessing or no queries in "
            "this batch must use unlocked preprocessing.");
      }
    }
  }

  vector<vector<int32_t>> query_tokens_storage;
  vector<ConstSpan<int32_t>> query_tokens(queries.size());
  if (pre_tokenized) {
    for (size_t i = 0; i < queries.size(); ++i) {
      auto tree_x_params =
          params[i]
              .searcher_specific_optional_parameters<TreeXOptionalParameters>();
      query_tokens[i] = tree_x_params->leaf_tokens_to_search();
    }
  } else if (has_unlocked_preprocessing) {
    for (size_t i : IndicesOf(queries)) {
      auto unlocked_preprocessing =
          params[i].unlocked_query_preprocessing_results<CentersToSearch>();
      query_tokens[i] = unlocked_preprocessing->centers_to_search();
    }
  } else {
    query_tokens_storage.resize(queries.size());
    vector<int32_t> max_centers_override(queries.size(), 0);
    bool override = false;

    for (int i = 0; i < queries.size(); ++i) {
      auto tree_x_params =
          params[i]
              .searcher_specific_optional_parameters<TreeXOptionalParameters>();
      if (tree_x_params)
        max_centers_override[i] =
            tree_x_params->num_partitions_to_search_override();
      if (max_centers_override[i] > 0) override = true;
    }

    if (override) {
      const auto* kmeans_tokenizer =
          down_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
      if (!kmeans_tokenizer)
        return InvalidArgumentError(
            "num_partitions_to_search_override is > 0, but the tokenizer is "
            "not a KMeansTreePartitioner.");

      SCANN_RETURN_IF_ERROR(
          kmeans_tokenizer->TokensForDatapointWithSpillingBatchedAndOverride(
              queries, max_centers_override,
              MakeMutableSpan(query_tokens_storage)));
    } else {
      SCANN_RETURN_IF_ERROR(
          query_tokenizer_->TokensForDatapointWithSpillingBatched(
              queries, MakeMutableSpan(query_tokens_storage)));
    }

    for (size_t i = 0; i < queries.size(); ++i) {
      query_tokens[i] = query_tokens_storage[i];
    }
  }

  if (!disjoint_leaf_partitions_ ||
      !tree_x_internal::SupportsLowLevelBatching(queries, params) ||
      tree_x_internal::RecursiveSize(query_tokens) < leaf_searchers_.size()) {
    return FindNeighborsPreTokenizedBatchedGenericImpl(queries, params,
                                                       query_tokens, results);
  } else {
    return FindNeighborsPreTokenizedBatchedOptimizedImpl(queries, params,
                                                         query_tokens, results);
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedBatchedGenericImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    ConstSpan<ConstSpan<int32_t>> query_tokens,
    MutableSpan<NNResultsVector> results) const {
  DCHECK_EQ(queries.size(), params.size());
  DCHECK_EQ(queries.size(), query_tokens.size());
  DCHECK_EQ(queries.size(), results.size());
  for (DatapointIndex i : IndicesOf(queries)) {
    if (params[i].pre_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      SCANN_RETURN_IF_ERROR(FindNeighborsPreTokenizedImpl(
          queries[i], params[i], query_tokens[i],
          TopNeighbors<float>(params[i].pre_reordering_num_neighbors()),
          &results[i]));
    }
  }
  return OkStatus();
}

namespace {

vector<std::vector<DatapointIndex>> InvertQueryTokens(
    ConstSpan<ConstSpan<int32_t>> query_tokens, size_t num_tokens) {
  vector<std::vector<DatapointIndex>> result(num_tokens);
  for (DatapointIndex query_index : IndicesOf(query_tokens)) {
    ConstSpan<int32_t> cur_query_tokens = query_tokens[query_index];
    for (int32_t qt : cur_query_tokens) {
      result[qt].push_back(query_index);
    }
  }
  return result;
}

size_t MaxQueriesPerPartition(
    ConstSpan<std::vector<DatapointIndex>> queries_by_partition) {
  size_t result = 0;
  for (auto& vec : queries_by_partition) {
    result = std::max(result, vec.size());
  }
  return result;
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedBatchedOptimizedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    ConstSpan<ConstSpan<int32_t>> query_tokens,
    MutableSpan<NNResultsVector> results) const {
  DCHECK(disjoint_leaf_partitions_);
  DCHECK(queries.IsDense());

  vector<std::vector<DatapointIndex>> queries_by_partition =
      InvertQueryTokens(query_tokens, leaf_searchers_.size());
  const size_t max_queries_per_partition =
      MaxQueriesPerPartition(queries_by_partition);
  vector<T> backing_storage;
  backing_storage.reserve(queries.dimensionality() * max_queries_per_partition);

  auto make_leaf_query_dataset =
      [&backing_storage, &queries](ConstSpan<DatapointIndex> query_idxs) {
        backing_storage.resize(0);
        for (DatapointIndex qi : query_idxs) {
          ConstSpan<T> values = queries[qi].values_slice();
          backing_storage.insert(backing_storage.end(), values.begin(),
                                 values.end());
        }
        return DenseDataset<T>(std::move(backing_storage), query_idxs.size());
      };

  vector<FastTopNeighbors<float>> top_ns;
  vector<FastTopNeighbors<float>::Mutator> mutators(params.size());
  top_ns.reserve(params.size());
  vector<shared_ptr<const SearcherSpecificOptionalParameters>>
      leaf_optional_params(queries.size());

  for (const auto& [query_idx, p] : Enumerate(params)) {
    top_ns.emplace_back(p.pre_reordering_num_neighbors(),
                        p.pre_reordering_epsilon());
    top_ns[query_idx].AcquireMutator(&mutators[query_idx]);
    TF_ASSIGN_OR_RETURN(
        leaf_optional_params[query_idx],
        CreateLeafOptionalParameters(queries[query_idx], params[query_idx]));
  }

  vector<NNResultsVector> leaf_results;
  leaf_results.reserve(max_queries_per_partition);
  for (auto [leaf_idx, query_idxs] : Enumerate(queries_by_partition)) {
    if (query_idxs.empty()) continue;
    DenseDataset<T> leaf_dataset = make_leaf_query_dataset(query_idxs);
    vector<SearchParameters> leaf_params =
        tree_x_internal::CreateParamsSubsetForLeaf<DatapointIndex>(
            params, mutators, leaf_optional_params, query_idxs);
    leaf_results.resize(0);
    leaf_results.resize(leaf_params.size());
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[leaf_idx]->FindNeighborsBatchedNoSortNoExactReorder(
            leaf_dataset, leaf_params, MakeMutableSpan(leaf_results)));
    backing_storage = leaf_dataset.ClearRecyclingDataVector();

    for (auto [local_query_idx, global_query_idx] : Enumerate(query_idxs)) {
      tree_x_internal::AddLeafResultsToTopN(
          *datapoints_by_token_[leaf_idx], 0.0f, 1.0f,
          leaf_results[local_query_idx], &mutators[global_query_idx]);
    }
  }

  mutators.clear();
  for (size_t i : IndicesOf(top_ns)) {
    top_ns[i].FinishUnsorted(&results[i]);
  }
  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::CheckReadyToQuery(
    const SearchParameters& params) const {
  if (leaf_searchers_.empty()) {
    return FailedPreconditionError("BuildLeafSearchers not called yet.");
  }

  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> query_tokens_storage;
  if (!PreTokenizationEnabled(tree_x_params) && query_tokenizer_ == nullptr) {
    return FailedPreconditionError(
        "Query tokenizer not set and pre-tokenization "
        "not enabled.");
  }

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::ValidateTokenList(ConstSpan<int32_t> token_list,
                                             bool check_oob) const {
  absl::flat_hash_set<int32_t> duplicate_checker;

  for (int32_t token : token_list) {
    if (duplicate_checker.find(token) == duplicate_checker.end()) {
      duplicate_checker.insert(token);
    } else {
      return InvalidArgumentError(
          absl::StrCat("Duplicate token:  ", token, "."));
    }

    if (token < 0) {
      return InvalidArgumentError("Tree-X hybrid tokens may not be negative.");
    }

    if (check_oob) {
      if (token >= datapoints_by_token_.size()) {
        return InvalidArgumentError(absl::StrCat("Token out of bounds (", token,
                                                 " vs. ",
                                                 datapoints_by_token_.size()));
      }
      if (token >= leaf_searchers_.size()) {
        return InvalidArgumentError(
            "Query token out of range of database tokens (got %d, "
            "expected in the range [0, %d).",
            static_cast<int>(token), static_cast<int>(leaf_searchers_.size()));
      }
    }
  }

  return OkStatus();
}

template <typename T>
StatusOr<shared_ptr<const SearcherSpecificOptionalParameters>>
TreeXHybridSMMD<T>::CreateLeafOptionalParameters(
    const DatapointPtr<T>& query,
    const SearchParameters& top_level_params) const {
  auto tree_x_params =
      top_level_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  const bool have_external_leaf_optional_params =
      tree_x_params && tree_x_params->all_leaf_optional_params();
  if (leaf_searcher_optional_parameter_creator_ &&
      have_external_leaf_optional_params) {
    return InvalidArgumentError(
        "Conflicting leaf searcher optional parameters.  Cannot have both "
        "external parameters from TreeXOptionalParameters and a "
        "LeafSearcherOptionalParameterCreator.");
  } else if (leaf_searcher_optional_parameter_creator_) {
    return leaf_searcher_optional_parameter_creator_
        ->CreateLeafSearcherOptionalParameters(query);
  } else if (have_external_leaf_optional_params) {
    return tree_x_params->all_leaf_optional_params();
  } else {
    return shared_ptr<const SearcherSpecificOptionalParameters>(nullptr);
  }
}

template <typename T>
template <typename TopN>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedImpl(
    const DatapointPtr<T>& query, const SearchParameters& params,
    ConstSpan<int32_t> query_tokens, TopN top_n,
    NNResultsVector* result) const {
  DCHECK(result);
  DCHECK(top_n.empty());

  if (query_tokens.empty()) {
    result->clear();
    return OkStatus();
  }

  if (params.restricts_enabled() && !PreTokenizationEnabled(params) &&
      this->reordering_enabled() &&
      std::isinf(params.pre_reordering_epsilon()) &&
      params.restrict_whitelist()->NumPointsWhitelisted() <=
          params.pre_reordering_num_neighbors()) {
    auto it = params.restrict_whitelist()->WhitelistedPointIterator();
    result->clear();
    for (; !it.Done(); it.Next()) {
      result->emplace_back(it.value(), params.pre_reordering_epsilon());
    }
    return OkStatus();
  }

  SCANN_RETURN_IF_ERROR(
      ValidateTokenList(query_tokens, query_tokenizer_ != nullptr));

  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  if (PreTokenizationEnabled(tree_x_params)) {
    DCHECK_EQ(tree_x_params->leaf_tokens_to_search().size(),
              query_tokens.size());
    DCHECK(std::equal(query_tokens.begin(), query_tokens.end(),
                      tree_x_params->leaf_tokens_to_search().begin()));
  }

  SearchParameters leaf_params;
  TF_ASSIGN_OR_RETURN(auto leaf_optional_params,
                      CreateLeafOptionalParameters(query, params));
  leaf_params.set_pre_reordering_num_neighbors(
      params.pre_reordering_num_neighbors());
  leaf_params.set_pre_reordering_epsilon(params.pre_reordering_epsilon());
  leaf_params.set_per_crowding_attribute_pre_reordering_num_neighbors(
      params.per_crowding_attribute_pre_reordering_num_neighbors());
  leaf_params.set_searcher_specific_optional_parameters(leaf_optional_params);

  if (query_tokens.size() == 1) {
    const auto token = query_tokens.front();
    if (token >= datapoints_by_token_.size()) {
      SCANN_LOG_NOOP(INFO, 10)
          << "With an empty partitioner, ignored a token of " << token
          << " that is >=" << datapoints_by_token_.size();
      return OkStatus();
    }

    TranslateGlobalToLeafLocalWhitelist(params, *datapoints_by_token_[token],
                                        &leaf_params);
    Status status = leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
        query, leaf_params, result);
    if (!status.ok()) return status;
    RemapToGlobalDatapointIndices(MakeMutableSpan(*result),
                                  *datapoints_by_token_[token]);
    return status;
  }

  if (disjoint_leaf_partitions_) {
    for (size_t i = 0; i < query_tokens.size(); ++i) {
      const int32_t token = query_tokens[i];
      if (token >= datapoints_by_token_.size()) {
        SCANN_LOG_NOOP(INFO, 10)
            << "With an empty partitioner, ignored a token of " << token
            << " that is >=" << datapoints_by_token_.size();
        continue;
      }

      TranslateGlobalToLeafLocalWhitelist(params, *datapoints_by_token_[token],
                                          &leaf_params);
      NNResultsVector leaf_results;
      SCANN_RETURN_IF_ERROR(
          leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
              query, leaf_params, &leaf_results));
      RemapToGlobalDatapointIndices(MakeMutableSpan(leaf_results),
                                    *datapoints_by_token_[token]);
      for (const auto& result : leaf_results) {
        top_n.push(result);
      }

      if (top_n.full()) {
        leaf_params.set_pre_reordering_epsilon(top_n.approx_bottom().second);
      }
    }
    *result = top_n.TakeUnsorted();
  } else {
    vector<NNResultsVector> leaf_results(query_tokens.size());

    for (size_t i = 0; i < query_tokens.size(); ++i) {
      const int32_t token = query_tokens[i];
      if (token >= datapoints_by_token_.size()) {
        SCANN_LOG_NOOP(INFO, 10)
            << "With an empty partitioner, ignored a token of " << token
            << " that is >=" << datapoints_by_token_.size();
        continue;
      }

      TranslateGlobalToLeafLocalWhitelist(params, *datapoints_by_token_[token],
                                          &leaf_params);
      Status status = leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
          query, leaf_params, &leaf_results[i]);
      if (!status.ok()) return status;
      RemapToGlobalDatapointIndices(MakeMutableSpan(leaf_results[i]),
                                    *datapoints_by_token_[token]);
    }

    MergeLeafResultsWithDuplicates(leaf_results, std::move(top_n), result);
  }

  return OkStatus();
}

template <typename T>
StatusOr<pair<int32_t, DatapointPtr<T>>>
TreeXHybridSMMD<T>::TokenizeAndMaybeResidualize(const DatapointPtr<T>& dptr,
                                                Datapoint<T>*) {
  DCHECK(database_tokenizer_);
  vector<int32_t> tokens_storage;
  SCANN_RETURN_IF_ERROR(database_tokenizer_->TokensForDatapointWithSpilling(
      dptr, &tokens_storage));
  if (tokens_storage.size() != 1) {
    return NotFoundError("Tokenizer must return exactly one token.");
  }
  return std::make_pair(tokens_storage[0], dptr);
}

template <typename T>
StatusOr<vector<pair<int32_t, DatapointPtr<T>>>>
TreeXHybridSMMD<T>::TokenizeAndMaybeResidualize(const TypedDataset<T>& dps,
                                                MutableSpan<Datapoint<T>*>) {
  vector<int32_t> token_storage(dps.size());
  SCANN_RETURN_IF_ERROR(
      database_tokenizer_->TokenForDatapointBatched(dps, &token_storage));
  vector<pair<int32_t, DatapointPtr<T>>> result(dps.size());
  for (size_t dp_idx : IndicesOf(dps)) {
    result[dp_idx] = {token_storage[dp_idx], dps[dp_idx]};
  }
  return result;
}

template <typename T>
StatusOr<SingleMachineFactoryOptions>
TreeXHybridSMMD<T>::ExtractSingleMachineFactoryOptions() {
  TF_ASSIGN_OR_RETURN(const int dataset_size,
                      UntypedSingleMachineSearcherBase::DatasetSize());
  auto int8_query_processor = std::dynamic_pointer_cast<
      const TreeScalarQuantizationPreprocessedQueryCreator>(
      leaf_searcher_optional_parameter_creator_);
  ConstSpan<float> int8_multipliers;
  if (int8_query_processor)
    int8_multipliers = int8_query_processor->inverse_multipliers();
  TF_ASSIGN_OR_RETURN(
      SingleMachineFactoryOptions leaf_opts,
      MergeAHLeafOptions(leaf_searchers_, datapoints_by_token_, dataset_size));

  TF_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<T>::ExtractSingleMachineFactoryOptions());
  int n_tokens = datapoints_by_token_.size();
  opts.datapoints_by_token =
      std::make_shared<vector<shared_ptr<vector<DatapointIndex>>>>(n_tokens);
  for (int i = 0; i < n_tokens; ++i) {
    (*opts.datapoints_by_token)[i] =
        std::make_shared<vector<DatapointIndex>>(*datapoints_by_token_[i]);
  }
  opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
  query_tokenizer_->CopyToProto(opts.serialized_partitioner.get());

  if (leaf_opts.ah_codebook != nullptr) {
    opts.ah_codebook = leaf_opts.ah_codebook;
    opts.hashed_dataset = leaf_opts.hashed_dataset;
  }
  if (leaf_opts.pre_quantized_fixed_point && !int8_multipliers.empty()) {
    opts.pre_quantized_fixed_point = make_shared<PreQuantizedFixedPoint>();
    opts.pre_quantized_fixed_point = leaf_opts.pre_quantized_fixed_point;
    opts.pre_quantized_fixed_point->multiplier_by_dimension =
        make_shared<vector<float>>(int8_multipliers.begin(),
                                   int8_multipliers.end());

    for (float& mult : *opts.pre_quantized_fixed_point->multiplier_by_dimension)
      mult = 1 / mult;
  }
  return opts;
}

template <typename T>
Status TreeXHybridSMMD<T>::PreprocessQueryIntoParamsUnlocked(
    const DatapointPtr<T>& query, SearchParameters& search_params) const {
  const auto& params =
      search_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> centers_to_search;
  if (params) {
    const auto* kmeans_tokenizer =
        down_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
    if (!kmeans_tokenizer)
      return InvalidArgumentError(
          "num_partitions_to_search_override is > 0, but the tokenizer is "
          "not a KMeansTreePartitioner.");

    int max_centers_override = params->num_partitions_to_search_override();
    SCANN_RETURN_IF_ERROR(
        kmeans_tokenizer->TokensForDatapointWithSpillingAndOverride(
            query, max_centers_override, &centers_to_search));
  } else {
    SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
        query, &centers_to_search));
  }

  search_params.set_unlocked_query_preprocessing_results(
      {make_unique<CentersToSearch>(std::move(centers_to_search))});
  return OkStatus();
}

template <typename T>
int TreeXHybridSMMD<T>::KmeanTokenizeDatabase( 
    shared_ptr<DenseDataset<T>> add_dataset,
    vector<std::vector<DatapointIndex>> &datapoints_by_token,
    ThreadPool* pool) {
  auto status_or_dt = database_tokenizer_->TokenizeDatabase(*add_dataset, pool);
  datapoints_by_token = std::move(status_or_dt.ValueOrDie());

  // TF_ASSIGN_OR_RETURN(datapoints_by_token,
  //                     database_tokenizer_->TokenizeDatabase(
  //                     *add_dataset, pool));
  if (datapoints_by_token.size() < database_tokenizer_->n_tokens()) {
    datapoints_by_token.resize(database_tokenizer_->n_tokens());
  }
  return 0;
}

template <typename T>
int TreeXHybridSMMD<T>::AddDataset(shared_ptr<DenseDataset<T>> raw_dataset,
    shared_ptr<DenseDataset<T>> add_dataset) {
  if (raw_dataset.get() == nullptr) {
    LOG(INFO) << "not reordering, dataset_ is nullptr";
    return -1;
  }

  size_t cur_datapoint_num = raw_dataset->size();
  size_t add_datapoint_num = add_dataset->size();
  size_t total_datapoint_num = cur_datapoint_num + add_datapoint_num;
  uint64_t dim = raw_dataset->Dimen();
  
  if (total_datapoint_num * dim > raw_dataset->Data().size()) {
    pthread_rwlock_wrlock(&dataset_rwlock_);
    raw_dataset->Data().resize(total_datapoint_num * dim *
        this->ExtendCoefficient(++dataset_extend_time_));
    pthread_rwlock_unlock(&dataset_rwlock_);
  }

  T *destin = raw_dataset->Data().data() + cur_datapoint_num * dim;
  T *source = add_dataset->Data().data();
  memcpy(destin, source, add_datapoint_num * dim * sizeof(T));
  for (int i = 0; i < add_datapoint_num; ++i) {
    raw_dataset->AddDocid("");
  }
  return 0;
}

template <typename T>
int TreeXHybridSMMD<T>::AddDatapointsForOneLeaf(
    std::vector<DatapointIndex> &datapoints_by_token, uint32_t leaf_id) {
  if (leaf_id >= datapoints_by_token_.size()) return -1;

  size_t n_token = datapoints_by_token.size();
  for (size_t i = 0; i < n_token; ++i) {
    datapoints_by_token[i] += num_datapoints_;
  }

  uint32_t n_leave_points = leaf_searchers_[leaf_id]->DatapointsNum();
  uint32_t idx_num = n_leave_points + datapoints_by_token.size();

  if (datapoints_by_token_[leaf_id]->size() < idx_num) {
    std::vector<DatapointIndex> *old_vec = datapoints_by_token_[leaf_id];
    std::vector<DatapointIndex> *new_vec = new std::vector<DatapointIndex>;
    new_vec->resize(1.5 * idx_num);
    memcpy(new_vec->data(), old_vec->data(), sizeof(DatapointIndex) * n_leave_points);
    datapoints_by_token_[leaf_id] = new_vec;
    AsyncWait(1000 * 100, [](std::vector<DatapointIndex> *del_vec) { delete del_vec; },
              old_vec);   // after 100s
  }

  DatapointIndex *destin = datapoints_by_token_[leaf_id]->data() + n_leave_points;
  DatapointIndex *source = datapoints_by_token.data();
  DatapointIndex add_num = datapoints_by_token.size();
  memcpy(destin, source, sizeof(DatapointIndex) * add_num);
  return 0;
}

template <typename T>
int TreeXHybridSMMD<T>::AddPackedDataset(
      const shared_ptr<DenseDataset<T>> add_dataset,
      vector<std::vector<DatapointIndex>> &datapoints_by_token,
      shared_ptr<ThreadPool> pool) {
  if (is_train_ == false) {
    LOG(INFO) << "TreeXHybridSMMD no train.";
    return -1;
  }

  int n_tokens = datapoints_by_token.size();
  for (int token = 0; token < n_tokens; ++token) {
    shared_ptr<TypedDataset<T>> dataset_partition(
        PartitionDataset(*add_dataset, datapoints_by_token[token]));
    shared_ptr<DenseDataset<uint8_t>> hashed_dataset =
        internal::HashLeafHelpers<T>::GetHashedDataset(dataset_partition, indexer_,
                                                       noise_shaping_threshold_, pool);
    AddDatapointsForOneLeaf(datapoints_by_token[token], token);
    leaf_searchers_[token]->AddSearcherPackedDataset(hashed_dataset);
  }
  num_datapoints_ += (add_dataset->size());
  return 0;
}

SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD();

}  // namespace research_scann
