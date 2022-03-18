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



#include "scann/base/single_machine_base.h"

#include <cstdint>
#include <typeinfo>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include <pthread.h>

namespace research_scann {

UntypedSingleMachineSearcherBase::~UntypedSingleMachineSearcherBase() {}

StatusOr<string_view> UntypedSingleMachineSearcherBase::GetDocid(
    DatapointIndex i) const {
  if (!docids_) {
    return FailedPreconditionError(
        "This SingleMachineSearcherBase instance does not have access to "
        "docids.  Did you call ReleaseDatasetAndDocids?");
  }

  const size_t n_docids = docids_->size();
  if (i >= n_docids) {
    return InvalidArgumentError("Datapoint index (%d) is >= dataset size (%d).",
                                i, n_docids);
  }

  return docids_->Get(i);
}

Status UntypedSingleMachineSearcherBase::set_docids(
    shared_ptr<const DocidCollectionInterface> docids) {
  if (dataset() || hashed_dataset()) {
    return FailedPreconditionError(
        "UntypedSingleMachineSearcherBase::set_docids may only be called "
        "on instances constructed using the constructor that does not accept "
        "a Dataset.");
  }

  if (docids_) {
    return FailedPreconditionError(
        "UntypedSingleMachineSearcherBase::set_docids may not be called if "
        "the docid array is not empty.  This can happen if set_docids has "
        "already been called on this instance, or if this instance was "
        "constructed using the constructor that takes a Dataset and then "
        "ReleaseDataset was called.");
  }

  docids_ = std::move(docids);
  return OkStatus();
}

void UntypedSingleMachineSearcherBase::SetUnspecifiedParametersToDefaults(
    SearchParameters* params) const {
  params->SetUnspecifiedParametersFrom(default_search_parameters_);
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    vector<int64_t> datapoint_index_to_crowding_attribute) {
  return EnableCrowding(std::make_shared<vector<int64_t>>(
      std::move(datapoint_index_to_crowding_attribute)));
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    shared_ptr<vector<int64_t>> datapoint_index_to_crowding_attribute) {
  SCANN_RET_CHECK(datapoint_index_to_crowding_attribute);
  if (!supports_crowding()) {
    return UnimplementedError("Crowding not supported for this searcher.");
  }
  if (crowding_enabled()) {
    return FailedPreconditionError("Crowding already enabled.");
  }
  SCANN_RETURN_IF_ERROR(
      EnableCrowdingImpl(*datapoint_index_to_crowding_attribute));
  datapoint_index_to_crowding_attribute_ =
      std::move(datapoint_index_to_crowding_attribute);
  return OkStatus();
}

StatusOr<DatapointIndex> UntypedSingleMachineSearcherBase::DatasetSize() const {
  if (dataset()) {
    return dataset()->size();
  } else if (hashed_dataset()) {
    return hashed_dataset()->size();
  } else if (docids_) {
    return docids_->size();
  } else {
    return FailedPreconditionError(
        "Dataset size is not known for this searcher.");
  }
}

bool UntypedSingleMachineSearcherBase::impl_needs_dataset() const {
  return true;
}

bool UntypedSingleMachineSearcherBase::impl_needs_hashed_dataset() const {
  return true;
}

DatapointIndex UntypedSingleMachineSearcherBase::optimal_batch_size() const {
  return 1;
}

UntypedSingleMachineSearcherBase::UntypedSingleMachineSearcherBase(
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon)
    : hashed_dataset_(hashed_dataset),
      default_search_parameters_(default_pre_reordering_num_neighbors,
                                 default_pre_reordering_epsilon,
                                 default_pre_reordering_num_neighbors,
                                 default_pre_reordering_epsilon) {
  if (default_pre_reordering_num_neighbors <= 0) {
    LOG(FATAL) << "default_pre_reordering_num_neighbors must be > 0, not "
               << default_pre_reordering_num_neighbors << ".";
  }

  if (std::isnan(default_pre_reordering_epsilon)) {
    LOG(FATAL) << "default_pre_reordering_epsilon must be non-NaN.";
  }
}

template <typename T>
SingleMachineSearcherBase<T>::SingleMachineSearcherBase(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon)
    : UntypedSingleMachineSearcherBase(hashed_dataset,
                                       default_pre_reordering_num_neighbors,
                                       default_pre_reordering_epsilon),
      dataset_(dataset) {
  TF_CHECK_OK(BaseInitImpl());
}

template <typename T>
Status SingleMachineSearcherBase<T>::BaseInitImpl() {
  if (hashed_dataset_ && dataset_ &&
      dataset_->size() != hashed_dataset_->size()) {
    return FailedPreconditionError(
        "If both dataset and hashed_dataset are provided, they must have the "
        "same size.");
  }

  if (dataset_) {
    docids_ = dataset_->docids();
  } else if (hashed_dataset_) {
    docids_ = hashed_dataset_->docids();
  } else {
    DCHECK(!docids_);
  }
  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::BaseInitFromDatasetAndConfig(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    const ScannConfig& config) {
  dataset_ = std::move(dataset);
  hashed_dataset_ = std::move(hashed_dataset);
  SCANN_RETURN_IF_ERROR(PopulateDefaultParameters(config));
  return BaseInitImpl();
}

template <typename T>
Status SingleMachineSearcherBase<T>::PopulateDefaultParameters(
    const ScannConfig& config) {
  GenericSearchParameters params;
  SCANN_RETURN_IF_ERROR(params.PopulateValuesFromScannConfig(config));
  const bool params_has_pre_norm =
      params.pre_reordering_dist->NormalizationRequired() != NONE;
  const bool params_has_exact_norm =
      params.reordering_dist->NormalizationRequired() != NONE;
  const bool dataset_has_norm =
      dataset_ && dataset_->normalization() !=
                      params.pre_reordering_dist->NormalizationRequired();
  if (params_has_pre_norm && !dataset_has_norm) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the pre-reordering distance "
        "measure.");
  }
  if (params_has_exact_norm && !dataset_has_norm) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the exact distance measure.");
  }
  const int32_t k = params.pre_reordering_num_neighbors;
  const float epsilon = params.pre_reordering_epsilon;
  default_search_parameters_ = SearchParameters(k, epsilon, k, epsilon);
  return OkStatus();
}

template <typename T>
SingleMachineSearcherBase<T>::SingleMachineSearcherBase(
    shared_ptr<const TypedDataset<T>> dataset,
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase(dataset, nullptr,
                                default_pre_reordering_num_neighbors,
                                default_pre_reordering_epsilon) {}

template <typename T>
SingleMachineSearcherBase<T>::~SingleMachineSearcherBase() {}

Status UntypedSingleMachineSearcherBase::SetMetadataGetter(
    shared_ptr<UntypedMetadataGetter> metadata_getter) {
  if (metadata_getter && metadata_getter->TypeTag() != this->TypeTag()) {
    return FailedPreconditionError(
        "SetMetadataGetter called with a MetadataGetter<%s>. Expected "
        "MetadataGetter<%s>.",
        TypeNameFromTag(metadata_getter->TypeTag()),
        TypeNameFromTag(this->TypeTag()));
  }
  metadata_getter_ = std::move(metadata_getter);
  return OkStatus();
}

template <typename T>
bool SingleMachineSearcherBase<T>::needs_dataset() const {
  return impl_needs_dataset() ||
         (reordering_enabled() && reordering_helper_->needs_dataset()) ||
         (metadata_enabled() && metadata_getter_->needs_dataset()) ||

         (dataset_ && mutator_outstanding_);
}

template <typename T>
StatusOr<SingleMachineFactoryOptions>
SingleMachineSearcherBase<T>::ExtractSingleMachineFactoryOptions() {
  SingleMachineFactoryOptions opts;

  opts.hashed_dataset =
      std::const_pointer_cast<DenseDataset<uint8_t>>(hashed_dataset_);

  opts.crowding_attributes = std::const_pointer_cast<vector<int64_t>>(
      datapoint_index_to_crowding_attribute_);
  opts.creation_timestamp = creation_timestamp_;
  if (reordering_helper_)
    reordering_helper_->AppendDataToSingleMachineFactoryOptions(&opts);
  return opts;
}

bool UntypedSingleMachineSearcherBase::needs_hashed_dataset() const {
  return impl_needs_hashed_dataset() ||

         (hashed_dataset_ && mutator_outstanding_);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighbors(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result, pthread_rwlock_t *rwlock) const {
  SCANN_RET_CHECK(query.IsFinite())
      << "Cannot query ScaNN with vectors that contain NaNs or infinity.";
  DCHECK(result);
  SCANN_RETURN_IF_ERROR(
      FindNeighborsNoSortNoExactReorder(query, params, result));

  if (reordering_helper_) {
    SCANN_RETURN_IF_ERROR(ReorderResults(query, params, result, rwlock));
  }

  return SortAndDropResults(result, params);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsNoSortNoExactReorder(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  bool reordering_enabled = exact_reordering_enabled();
  SCANN_RETURN_IF_ERROR(params.Validate(reordering_enabled));
  if (!this->supports_crowding() && params.pre_reordering_crowding_enabled()) {
    return InvalidArgumentError(
        std::string(
            "Crowding is enabled but not supported for searchers of type ") +
        typeid(*this).name() + ".");
  }
  if (!this->crowding_enabled() && params.crowding_enabled()) {
    return InvalidArgumentError(
        "Crowding is enabled for query but not enabled in searcher.");
  }

  if (dataset() && !dataset()->empty() &&
      query.dimensionality() != dataset()->dimensionality()) {
    return FailedPreconditionError(
        StrFormat("Query dimensionality (%u) does not match database "
                  "dimensionality (%u)",
                  static_cast<uint64_t>(query.dimensionality()),
                  static_cast<uint64_t>(dataset()->dimensionality())));
  }

  return FindNeighborsImpl(query, params, result);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatched(
    const TypedDataset<T>& queries, MutableSpan<NNResultsVector> result,
    pthread_rwlock_t *rwlock) const {
  vector<SearchParameters> params(queries.size());
  for (auto& p : params) {
    p.SetUnspecifiedParametersFrom(default_search_parameters_);
  }
  return FindNeighborsBatched(queries, params, result, rwlock);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatched(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results, pthread_rwlock_t *rwlock) const {
  SCANN_RETURN_IF_ERROR(
      FindNeighborsBatchedNoSortNoExactReorder(queries, params, results));

  if (reordering_helper_) {
    for (DatapointIndex i = 0; i < queries.size(); ++i) {
      SCANN_RETURN_IF_ERROR(ReorderResults(queries[i], params[i], &results[i], rwlock));
    }
  }

  for (DatapointIndex i = 0; i < results.size(); ++i) {
    SCANN_RETURN_IF_ERROR(SortAndDropResults(&results[i], params[i]));
  }

  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedNoSortNoExactReorder(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if (queries.size() != params.size()) {
    return InvalidArgumentError(
        "queries.size != params.size in FindNeighbors batched (%d vs. %d).",
        queries.size(), params.size());
  }

  if (queries.size() != results.size()) {
    return InvalidArgumentError(
        "queries.size != results.size in FindNeighbors batched (%d vs. %d).",
        queries.size(), results.size());
  }
  for (auto [query_idx, param] : Enumerate(params)) {
    if (!this->supports_crowding() && param.pre_reordering_crowding_enabled()) {
      return InvalidArgumentError(
          absl::Substitute("Crowding is enabled for query (index $0) but not "
                           "supported for searchers of type $1.",
                           query_idx, typeid(*this).name()));
    }
    if (!this->crowding_enabled() && param.crowding_enabled()) {
      return InvalidArgumentError(
          absl::Substitute("Crowding is enabled for query (index $0) but not "
                           "enabled in searcher.",
                           query_idx));
    }
  }

  bool reordering_enabled = exact_reordering_enabled();
  for (const SearchParameters& p : params) {
    SCANN_RETURN_IF_ERROR(p.Validate(reordering_enabled));
  }

  if (dataset() && !dataset()->empty() &&
      queries.dimensionality() != dataset()->dimensionality()) {
    return FailedPreconditionError(
        "Query dimensionality (%u) does not match database dimensionality (%u)",
        queries.dimensionality(), dataset()->dimensionality());
  }

  return FindNeighborsBatchedImpl(queries, params, results);
}

template <typename T>
Status SingleMachineSearcherBase<T>::GetNeighborProto(
    const pair<DatapointIndex, float> neighbor, const DatapointPtr<T>& query,
    NearestNeighbors::Neighbor* result) const {
  SCANN_RETURN_IF_ERROR(GetNeighborProtoNoMetadata(neighbor, query, result));
  if (!metadata_enabled()) return OkStatus();

  Status status = metadata_getter()->GetMetadata(
      dataset(), query, neighbor.first, result->mutable_metadata());
  if (!status.ok()) result->Clear();
  return status;
}

template <typename T>
Status SingleMachineSearcherBase<T>::GetNeighborProtoNoMetadata(
    const pair<DatapointIndex, float> neighbor, const DatapointPtr<T>& query,
    NearestNeighbors::Neighbor* result) const {
  DCHECK(result);
  result->Clear();
  TF_ASSIGN_OR_RETURN(auto docid, GetDocid(neighbor.first));
  result->set_docid(std::string(docid));
  result->set_distance(neighbor.second);
  if (crowding_enabled()) {
    result->set_crowding_attribute(
        (*datapoint_index_to_crowding_attribute_)[neighbor.first]);
  }
  return OkStatus();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseDataset() {
  if (needs_dataset()) {
    LOG(FATAL) << "Cannot release dataset for this instance.";
    return;
  }

  if (!dataset_) return;

  if (hashed_dataset()) {
    DCHECK_EQ(docids_.get(), dataset_->docids().get());
    docids_ = hashed_dataset_->docids();
  }

  dataset_.reset();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseHashedDataset() {
  if (!hashed_dataset_) return;
  hashed_dataset_.reset();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseDatasetAndDocids() {
  if (needs_dataset()) {
    LOG(FATAL) << "Cannot release dataset for this instance.";
    return;
  }

  dataset_.reset();
  docids_.reset();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  DCHECK_EQ(queries.size(), params.size());
  DCHECK_EQ(queries.size(), results.size());
  for (DatapointIndex i = 0; i < queries.size(); ++i) {
    SCANN_RETURN_IF_ERROR(
        FindNeighborsImpl(queries[i], params[i], &results[i]));
  }

  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::ReorderResults(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result, pthread_rwlock_t *rwlock) const {
  if (rwlock != nullptr) {
    pthread_rwlock_rdlock(rwlock);
  }
  if (params.post_reordering_num_neighbors() == 1) {
    TF_ASSIGN_OR_RETURN(
        auto top1,
        reordering_helper_->ComputeTop1ReorderingDistance(query, result));
    if (!result->empty() && top1.second < params.post_reordering_epsilon() &&
        top1.first != kInvalidDatapointIndex) {
      result->resize(1);
      result->at(0) = top1;
    } else {
      result->resize(0);
    }
  } else {
    SCANN_RETURN_IF_ERROR(
        reordering_helper_->ComputeDistancesForReordering(query, result));
  }
  if (rwlock != nullptr) {
    pthread_rwlock_unlock(rwlock);
  }
  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::SortAndDropResults(
    NNResultsVector* result, const SearchParameters& params) const {
  if (reordering_enabled()) {
    if (params.post_reordering_num_neighbors() == 1) {
      return OkStatus();
    }

    if (params.post_reordering_epsilon() < numeric_limits<float>::infinity()) {
      auto it = std::partition(
          result->begin(), result->end(),
          [&params](const pair<DatapointIndex, float>& arg) {
            return arg.second <= params.post_reordering_epsilon();
          });
      const size_t new_size = it - result->begin();
      result->resize(new_size);
    }

    if (params.post_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      RemoveNeighborsPastLimit(params.post_reordering_num_neighbors(), result);
    }
  }

  if (params.sort_results()) {
    ZipSortBranchOptimized(DistanceComparatorBranchOptimized(), result->begin(),
                           result->end());
  }
  return OkStatus();
}

template <typename T>
bool SingleMachineSearcherBase<T>::fixed_point_reordering_enabled() const {
  return (reordering_helper_ &&
          absl::StartsWith(reordering_helper_->name(), "FixedPoint"));
}

SCANN_INSTANTIATE_TYPED_CLASS(, SingleMachineSearcherBase);

}  // namespace research_scann
