/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_index_ivfpq_gpu.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexShards.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <set>
#include <vector>

#include "bitmap.h"
#include "gamma_api.h"
#include "gamma_gpu_cloner.h"
#include "gamma_index_ivfpq.h"

using std::string;
using std::vector;

namespace tig_gamma {
namespace gamma_gpu {

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

namespace {
const int kMaxBatch = 200;  // max search batch num
const int kMaxReqNum = 200;
const char *kDelim = "\001";
}  // namespace

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue() : mutex_(), condvar_(), queue_() {}

  void Put(const T &task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(task);
    }
    condvar_.notify_all();
  }

  T Take() {
    std::unique_lock<std::mutex> lock(mutex_);
    condvar_.wait(lock, [this] { return !queue_.empty(); });
    assert(!queue_.empty());
    T front(queue_.front());
    queue_.pop_front();

    return front;
  }

  T TakeBatch(vector<T> &vec, int &n) {
    std::unique_lock<std::mutex> lock(mutex_);
    condvar_.wait(lock, [this] { return !queue_.empty(); });
    assert(!queue_.empty());

    int i = 0;
    while (!queue_.empty() && i < kMaxBatch) {
      T front(queue_.front());
      queue_.pop_front();
      vec[i++] = front;
    }
    n = i;

    return nullptr;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  BlockingQueue(const BlockingQueue &);
  BlockingQueue &operator=(const BlockingQueue &);

 private:
  mutable std::mutex mutex_;
  std::condition_variable condvar_;
  std::list<T> queue_;
};

class GPUItem {
 public:
  GPUItem(int n, const float *x, int k, float *dis, long *label, int nprobe)
      : x_(x) {
    n_ = n;
    k_ = k;
    dis_ = dis;
    label_ = label;
    nprobe_ = nprobe;
    done_ = false;
    batch_size = 1;
  }

  void Notify() {
    done_ = true;
    cv_.notify_one();
  }

  int WaitForDone() {
    std::unique_lock<std::mutex> lck(mtx_);
    while (not done_) {
      cv_.wait_for(lck, std::chrono::seconds(1),
                   [this]() -> bool { return done_; });
    }
    return 0;
  }

  int n_;
  const float *x_;
  float *dis_;
  int k_;
  long *label_;
  int nprobe_;

  int batch_size;  // for perfomance test

 private:
  std::condition_variable cv_;
  std::mutex mtx_;
  bool done_;
};

GammaIVFPQGPUIndex::GammaIVFPQGPUIndex(size_t d, size_t nlist, size_t M,
                                       size_t nbits_per_idx,
                                       const char *docids_bitmap,
                                       RawVector<float> *raw_vec,
                                       GammaCounters *counters)
    : GammaIndex(d, docids_bitmap) {
  this->nlist_ = nlist;
  this->M_ = M;
  this->nbits_per_idx_ = nbits_per_idx;
  this->nprobe_ = 20;
  indexed_vec_count_ = 0;
  search_idx_ = 0;
  b_exited_ = false;
  use_standard_resource_ = true;
  is_indexed_ = false;
  this->SetRawVectorFloat(raw_vec);
  faiss::IndexFlatL2 *coarse_quantizer = new faiss::IndexFlatL2(d);

  gpu_index_ = nullptr;
  cpu_index_ = new GammaIVFPQIndex(coarse_quantizer, d, nlist, M, nbits_per_idx,
                                   docids_bitmap, raw_vec, counters);
#ifdef PERFORMANCE_TESTING
  search_count_ = 0;
#endif
}

GammaIVFPQGPUIndex::~GammaIVFPQGPUIndex() {
  std::lock_guard<std::mutex> lock(indexing_mutex_);
  b_exited_ = true;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  if (cpu_index_) {
    delete cpu_index_;
    cpu_index_ = nullptr;
  }

  if (gpu_index_) {
    delete gpu_index_;
    gpu_index_ = nullptr;
  }

  for (auto &resource : resources_) {
    delete resource;
    resource = nullptr;
  }
  resources_.clear();
}

faiss::Index *GammaIVFPQGPUIndex::CreateGPUIndex() {
  int ngpus = faiss::gpu::getNumDevices();

  vector<int> devs;
  for (int i = 0; i < ngpus; ++i) {
    devs.push_back(i);
  }

  if (not is_indexed_) {
    if (not use_standard_resource_) {
      GammaMemManager manager;
      tmp_mem_num_ = manager.Init(ngpus);
      LOG(INFO) << "Resource num [" << tmp_mem_num_ << "]";
    }

    for (int i : devs) {
      if (use_standard_resource_) {
        auto res = new faiss::gpu::StandardGpuResources;
        res->initializeForDevice(i);
        res->setTempMemory((size_t)1536 * 1024 * 1024);  // 1.5 GiB
        resources_.push_back(res);
      } else {
        auto res = new faiss::gpu::GammaGpuResources;
        res->initializeForDevice(i);
        resources_.push_back(res);
      }
    }
  }

  faiss::gpu::GpuMultipleClonerOptions *options =
      new faiss::gpu::GpuMultipleClonerOptions();

  options->indicesOptions = faiss::gpu::INDICES_64_BIT;
  options->useFloat16CoarseQuantizer = false;
  options->useFloat16 = true;
  options->usePrecomputed = false;
  options->reserveVecs = 0;
  options->storeTransposed = true;
  options->verbose = true;

  // shard the index across GPUs
  options->shard = true;
  options->shard_type = 1;

  std::lock_guard<std::mutex> lock(cpu_mutex_);
  faiss::Index *gpu_index =
      gamma_index_cpu_to_gpu_multiple(resources_, devs, cpu_index_, options);

  delete options;
  return gpu_index;
}

int GammaIVFPQGPUIndex::CreateSearchThread() {
  auto func_search = std::bind(&GammaIVFPQGPUIndex::GPUThread, this);

  gpu_threads_.push_back(std::thread(func_search));
  gpu_threads_[0].detach();

  return 0;
}

int GammaIVFPQGPUIndex::Indexing() {
  std::lock_guard<std::mutex> lock(indexing_mutex_);
  int vectors_count = raw_vec_->GetVectorNum();
  if (vectors_count < 8192) {
    LOG(ERROR) << "vector total count [" << vectors_count
               << "] less then 8192, failed!";
    return -1;
  }

  LOG(INFO) << "GPU indexing";

  faiss::Index *index = nullptr;

  if (not is_indexed_) {
    int num = vectors_count > 100000 ? 100000 : vectors_count;
    ScopeVector<float> scope_vec;
    raw_vec_->GetVectorHeader(0, num, scope_vec);
    int raw_d = raw_vec_->GetDimension();
    float *train_vec = nullptr;

    if (d_ > raw_d) {
      float *vec = new float[num * d_];
      ConvertVectorDim(num, raw_d, d_, scope_vec.Get(), vec);
      train_vec = vec;
    } else {
      train_vec = const_cast<float *>(scope_vec.Get());
    }
    cpu_index_->train(num, train_vec);

    if (d_ > raw_d) {
      delete train_vec;
    }

    while (indexed_vec_count_ < vectors_count) {
      AddRTVecsToIndex();
    }
    gpu_index_ = CreateGPUIndex();
    CreateSearchThread();
    is_indexed_ = true;
    LOG(INFO) << "GPU indexed.";
    return 0;
  }

  index = CreateGPUIndex();

  auto old_index = gpu_index_;
  gpu_index_ = index;

  std::this_thread::sleep_for(std::chrono::seconds(2));
  delete old_index;
  LOG(INFO) << "GPU indexed.";
  return 0;
}

int GammaIVFPQGPUIndex::AddRTVecsToIndex() {
  std::lock_guard<std::mutex> lock(cpu_mutex_);
  int ret = cpu_index_->AddRTVecsToIndex();
  indexed_vec_count_ = cpu_index_->indexed_vec_count_;
  return ret;
}

int GammaIVFPQGPUIndex::Update(int doc_id, const float *vec) {
  std::lock_guard<std::mutex> lock(indexing_mutex_);
  return cpu_index_->Update(doc_id, vec);
}

int GammaIVFPQGPUIndex::Search(const VectorQuery *query,
                               GammaSearchCondition *condition,
                               VectorResult &result) {
  int raw_d = raw_vec_->GetDimension();
  if (gpu_threads_.size() == 0) {
    LOG(ERROR) << "gpu index not indexed!";
    return -1;
  }

  float *xq = reinterpret_cast<float *>(query->value->value);
  int n = query->value->len / (raw_d * sizeof(float));
  if (n > kMaxReqNum) {
    LOG(ERROR) << "req num [" << n << "] should not larger than [" << kMaxReqNum
               << "]";
    return -1;
  }

#ifdef PERFORMANCE_TESTING
  condition->Perf("search prepare");
#endif

  float *vec_q = nullptr;

  if (d_ > raw_d) {
    float *vec = new float[n * d_];
    ConvertVectorDim(n, raw_d, d_, xq, vec);
    vec_q = vec;
  } else {
    vec_q = xq;
  }
  GPUSearch(n, vec_q, condition->topn, result.dists, result.docids, condition);

  if (d_ > raw_d) {
    delete vec_q;
  }

#ifdef PERFORMANCE_TESTING
  condition->Perf("GPU search");
#endif

  for (int i = 0; i < n; i++) {
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

#ifdef PERFORMANCE_TESTING
  condition->Perf("reorder");
#endif
  return 0;
}

int GammaIVFPQGPUIndex::Delete(int doc_id) {
  std::lock_guard<std::mutex> lock(indexing_mutex_);
  return cpu_index_->Delete(doc_id);
}

long GammaIVFPQGPUIndex::GetTotalMemBytes() {
  return cpu_index_->GetTotalMemBytes();
}

int GammaIVFPQGPUIndex::Dump(const string &dir, int max_vid) {
  return cpu_index_->Dump(dir, max_vid);
}

int GammaIVFPQGPUIndex::Load(const vector<string> &index_dirs) {
  int ret = cpu_index_->Load(index_dirs);
  return ret;
}

int GammaIVFPQGPUIndex::GPUThread() {
  GammaMemManager manager;
  std::thread::id tid = std::this_thread::get_id();
  float *xx = new float[kMaxBatch * d_ * kMaxReqNum];
  size_t max_recallnum = (size_t)faiss::gpu::getMaxKSelection();
  long *label = new long[kMaxBatch * max_recallnum * kMaxReqNum];
  float *dis = new float[kMaxBatch * max_recallnum * kMaxReqNum];

  while (not b_exited_) {
    int size = 0;
    GPUItem *items[kMaxBatch];

    while (size == 0 && not b_exited_) {
      size = id_queue_.wait_dequeue_bulk_timed(items, kMaxBatch, 1000);
    }

    if (size > 1) {
      std::map<int, std::vector<int>> nprobe_map;
      for (int i = 0; i < size; ++i) {
        nprobe_map[items[i]->nprobe_].push_back(i);
      }

      for (auto nprobe_ids : nprobe_map) {
        int recallnum = 0, cur = 0, total = 0;
        for (size_t j = 0; j < nprobe_ids.second.size(); ++j) {
          recallnum = std::max(recallnum, items[nprobe_ids.second[j]]->k_);
          total += items[nprobe_ids.second[j]]->n_;
          memcpy(xx + cur, items[nprobe_ids.second[j]]->x_,
                 d_ * sizeof(float) * items[nprobe_ids.second[j]]->n_);
          cur += d_ * items[nprobe_ids.second[j]]->n_;
        }

        int ngpus = faiss::gpu::getNumDevices();
        if (ngpus > 1) {
          auto indexShards = dynamic_cast<faiss::IndexShards *>(gpu_index_);
          if (indexShards != nullptr) {
            for (int j = 0; j < indexShards->count(); ++j) {
              auto ivfpq =
                dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(indexShards->at(j));
              ivfpq->setNumProbes(nprobe_ids.first);
            }
          }
        } else {
          auto ivfpq = dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(gpu_index_);
          if (ivfpq != nullptr)
            ivfpq->setNumProbes(nprobe_ids.first);
        }
        gpu_index_->search(total, xx, recallnum, dis, label);

        cur = 0;
        for (size_t j = 0; j < nprobe_ids.second.size(); ++j) {
          memcpy(items[nprobe_ids.second[j]]->dis_, dis + cur,
                 recallnum * sizeof(float) * items[nprobe_ids.second[j]]->n_);
          memcpy(items[nprobe_ids.second[j]]->label_, label + cur,
                 recallnum * sizeof(long) * items[nprobe_ids.second[j]]->n_);
          items[nprobe_ids.second[j]]->batch_size = nprobe_ids.second.size();
          cur += recallnum * items[nprobe_ids.second[j]]->n_;
          items[nprobe_ids.second[j]]->Notify();
        }
      }
    } else if (size == 1) {
      int ngpus = faiss::gpu::getNumDevices();
      if (ngpus > 1) {
        auto indexShards = dynamic_cast<faiss::IndexShards *>(gpu_index_);
        if (indexShards != nullptr) {
          for (int j = 0; j < indexShards->count(); ++j) {
            auto ivfpq =
              dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(indexShards->at(j));
            ivfpq->setNumProbes(items[0]->nprobe_);
          }
        }
      } else {
        auto ivfpq = dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(gpu_index_);
        if (ivfpq != nullptr)
          ivfpq->setNumProbes(items[0]->nprobe_);
      }
      gpu_index_->search(items[0]->n_, items[0]->x_, items[0]->k_,
                         items[0]->dis_, items[0]->label_);
      items[0]->batch_size = size;
      items[0]->Notify();
    }
  }

  if (not use_standard_resource_) {
    manager.ReturnMem(tid);
  }

  delete[] xx;
  delete[] label;
  delete[] dis;
  LOG(INFO) << "thread exit";
  return 0;
}

namespace {

int ParseFilters(GammaSearchCondition *condition,
                 vector<string> &range_filter_names,
                 vector<enum DataType> &range_filter_types,
                 vector<string> &term_filter_names,
                 vector<enum DataType> &term_filter_types,
                 vector<vector<string>> all_term_items) {
  for (int i = 0; i < condition->range_filters_num; ++i) {
    auto range = condition->range_filters[i];
    range_filter_names[i] = string(range->field->value, range->field->len);

    enum DataType type;
    if (condition->profile->GetFieldType(range_filter_names[i], type)) {
      LOG(ERROR) << "Can't get " << range_filter_names[i] << " data type";
      return -1;
    }

    if (type == DataType::STRING) {
      LOG(ERROR) << "Wrong type: " << type << ", " << range_filter_names[i]
                 << " can't be range filter";
      return -1;
    }
    range_filter_types[i] = type;
  }

  for (int i = 0; i < condition->term_filters_num; ++i) {
    auto term = condition->term_filters[i];

    term_filter_names[i] = string(term->field->value, term->field->len);

    enum DataType type;
    if (condition->profile->GetFieldType(term_filter_names[i], type)) {
      LOG(ERROR) << "Can't get " << term_filter_names[i] << " data type";
      return -1;
    }

    if (type != DataType::STRING) {
      LOG(ERROR) << "Wrong type: " << type << ", " << term_filter_names[i]
                 << " can't be term filter";
      return -1;
    }

    term_filter_types[i] = type;
    vector<string> term_items =
        utils::split(string(term->value->value, term->value->len), kDelim);
    all_term_items[i] = term_items;
  }
  return 0;
}

template <class T>
bool IsInRange(Profile *profile, RangeFilter *range, long docid,
               string &field_name) {
  T value = 0;
  profile->GetField<T>(docid, field_name, value);
  T lower_value, upper_value;
  memcpy(&lower_value, range->lower_value->value, range->lower_value->len);
  memcpy(&upper_value, range->upper_value->value, range->upper_value->len);

  if (range->include_lower != 0 && range->include_upper != 0) {
    if (value >= lower_value && value <= upper_value) return true;
  } else if (range->include_lower != 0 && range->include_upper == 0) {
    if (value >= lower_value && value < upper_value) return true;
  } else if (range->include_lower == 0 && range->include_upper != 0) {
    if (value > lower_value && value <= upper_value) return true;
  } else {
    if (value > lower_value && value < upper_value) return true;
  }
  return false;
}

bool FilteredByRangeFilter(GammaSearchCondition *condition,
                           vector<string> &range_filter_names,
                           vector<enum DataType> &range_filter_types,
                           long docid) {
  for (int i = 0; i < condition->range_filters_num; ++i) {
    auto range = condition->range_filters[i];

    if (range_filter_types[i] == DataType::INT) {
      if (!IsInRange<int>(condition->profile, range, docid,
                          range_filter_names[i]))
        return true;
    } else if (range_filter_types[i] == DataType::LONG) {
      if (!IsInRange<long>(condition->profile, range, docid,
                           range_filter_names[i]))
        return true;
    } else if (range_filter_types[i] == DataType::FLOAT) {
      if (!IsInRange<float>(condition->profile, range, docid,
                            range_filter_names[i]))
        return true;
    } else {
      if (!IsInRange<double>(condition->profile, range, docid,
                             range_filter_names[i]))
        return true;
    }
  }
  return false;
}

bool FilteredByTermFilter(GammaSearchCondition *condition,
                          vector<string> &term_filter_names,
                          vector<enum DataType> &term_filter_types,
                          vector<vector<string>> &all_term_items, long docid) {
  for (int i = 0; i < condition->term_filters_num; ++i) {
    auto term = condition->term_filters[i];

    char *field_value;
    int len = condition->profile->GetFieldString(docid, term_filter_names[i],
                                                 &field_value);
    vector<string> field_items;
    if (len >= 0) field_items = utils::split(string(field_value, len), kDelim);

    bool all_in_field_items;
    if (term->is_union)
      all_in_field_items = false;
    else
      all_in_field_items = true;

    for (auto term_item : all_term_items[i]) {
      bool in_field_items = false;
      for (size_t j = 0; j < field_items.size(); j++) {
        if (term_item == field_items[j]) {
          in_field_items = true;
          break;
        }
      }
      if (term->is_union)
        all_in_field_items |= in_field_items;
      else
        all_in_field_items &= in_field_items;
    }
    if (!all_in_field_items) return true;
  }
  return false;
};

}  // namespace

int GammaIVFPQGPUIndex::GPUSearch(int n, const float *x, int k,
                                  float *distances, long *labels,
                                  GammaSearchCondition *condition) {
  size_t recall_num = (size_t)condition->recall_num;

  size_t max_recallnum = (size_t)faiss::gpu::getMaxKSelection();
  if (recall_num > max_recallnum) {
    LOG(WARNING) << "recall_num should less than [" << max_recallnum << "]";
    recall_num = max_recallnum;
  }
  int nprobe = this->nprobe_;
  if (condition->nprobe > 0 && (size_t)condition->nprobe <= this->nlist_ &&
      (size_t)condition->nprobe <= max_recallnum) {
    nprobe = condition->nprobe;
  } else {
    LOG(WARNING) << "Error nprobe for search, so using default value:"
                 << this->nprobe_;
  }

  vector<float> D(n * max_recallnum);
  vector<long> I(n * max_recallnum);

#ifdef PERFORMANCE_TESTING
  condition->Perf("GPUSearch prepare");
#endif
  GPUItem *item = new GPUItem(n, x, recall_num, D.data(), I.data(), nprobe);

  id_queue_.enqueue(item);

  item->WaitForDone();

  delete item;

#ifdef PERFORMANCE_TESTING
  condition->Perf("GPU thread");
#endif

  bool right_filter = true;
  vector<string> range_filter_names(condition->range_filters_num);
  vector<enum DataType> range_filter_types(condition->range_filters_num);

  vector<string> term_filter_names(condition->term_filters_num);
  vector<enum DataType> term_filter_types(condition->term_filters_num);
  vector<vector<string>> all_term_items(condition->term_filters_num);

  if (ParseFilters(condition, range_filter_names, range_filter_types,
                   term_filter_names, term_filter_types, all_term_items)) {
    right_filter = false;
  }

  // set filter
  auto is_filterable = [&](long docid) -> bool {
    return bitmap::test(docids_bitmap_, docid) ||
           (right_filter &&
            (FilteredByRangeFilter(condition, range_filter_names,
                                   range_filter_types, docid) ||
             FilteredByTermFilter(condition, term_filter_names,
                                  term_filter_types, all_term_items, docid)));
  };

  using HeapForIP = faiss::CMin<float, idx_t>;
  using HeapForL2 = faiss::CMax<float, idx_t>;

  auto init_result = [&](int topk, float *simi, idx_t *idxi) {
    if (condition->metric_type == DistanceMetricType::InnerProduct) {
      faiss::heap_heapify<HeapForIP>(topk, simi, idxi);
    } else {
      faiss::heap_heapify<HeapForL2>(topk, simi, idxi);
    }
  };

  auto reorder_result = [&](int topk, float *simi, idx_t *idxi) {
    if (condition->metric_type == DistanceMetricType::InnerProduct) {
      faiss::heap_reorder<HeapForIP>(topk, simi, idxi);
    } else {
      faiss::heap_reorder<HeapForL2>(topk, simi, idxi);
    }
  };

  int raw_d = raw_vec_->GetDimension();

  std::function<void(const float **)> compute_vec;

  if (condition->has_rank) {
    compute_vec = [&](const float **vecs) {
      for (int i = 0; i < n; ++i) {
        const float *xi = x + i * d_;  // query

        float *simi = distances + i * k;
        long *idxi = labels + i * k;
        init_result(k, simi, idxi);

        for (size_t j = 0; j < recall_num; ++j) {
          long vid = I[i * recall_num + j];
          if (vid < 0) {
            continue;
          }

          int docid = raw_vec_->vid_mgr_->VID2DocID(vid);
          if (is_filterable(docid)) {
            continue;
          }

          float dist = -1;
          if (condition->metric_type == DistanceMetricType::InnerProduct) {
            dist =
                faiss::fvec_inner_product(xi, vecs[i * recall_num + j], raw_d);
          } else {
            dist = faiss::fvec_L2sqr(xi, vecs[i * recall_num + j], raw_d);
          }

          if (((condition->min_dist >= 0 && dist >= condition->min_dist) &&
               (condition->max_dist >= 0 && dist <= condition->max_dist)) ||
              (condition->min_dist == -1 && condition->max_dist == -1)) {
            if (condition->metric_type == DistanceMetricType::InnerProduct) {
              if (HeapForIP::cmp(simi[0], dist)) {
                faiss::heap_pop<HeapForIP>(k, simi, idxi);
                faiss::heap_push<HeapForIP>(k, simi, idxi, dist, vid);
              }
            } else {
              if (HeapForL2::cmp(simi[0], dist)) {
                faiss::heap_pop<HeapForL2>(k, simi, idxi);
                faiss::heap_push<HeapForL2>(k, simi, idxi, dist, vid);
              }
            }
          }
        }

        if (condition->sort_by_docid) {
          vector<std::pair<long, float>> id_sim_pairs(k);
          for (int z = 0; z < k; ++z) {
            id_sim_pairs[z] = std::move(std::make_pair(idxi[z], simi[z]));
          }
          std::sort(id_sim_pairs.begin(), id_sim_pairs.end());
          for (int z = 0; z < k; ++z) {
            idxi[z] = id_sim_pairs[z].first;
            simi[z] = id_sim_pairs[z].second;
          }
        } else {
          reorder_result(k, simi, idxi);
        }
      }  // parallel
    };
  } else {
    compute_vec = [&](const float **vecs) {
      for (int i = 0; i < n; ++i) {
        float *simi = distances + i * k;
        long *idxi = labels + i * k;
        int idx = 0;
        memset(simi, -1, sizeof(float) * k);
        memset(idxi, -1, sizeof(long) * k);

        for (size_t j = 0; j < recall_num; ++j) {
          long vid = I[i * recall_num + j];
          if (vid < 0) {
            continue;
          }

          int docid = raw_vec_->vid_mgr_->VID2DocID(vid);
          if (is_filterable(docid)) {
            continue;
          }

          float dist = D[i * recall_num + j];

          if (((condition->min_dist >= 0 && dist >= condition->min_dist) &&
               (condition->max_dist >= 0 && dist <= condition->max_dist)) ||
              (condition->min_dist == -1 && condition->max_dist == -1)) {
            simi[idx] = dist;
            idxi[idx] = vid;
            idx++;
            if (idx >= k) break;
          }
        }
      }
    };
  }

  std::function<void()> compute_dis;

  if (condition->has_rank) {
    // calculate inner product for selected possible vectors
    compute_dis = [&]() {
      ScopeVectors<float> scope_vecs(recall_num * n);
      raw_vec_->Gets(recall_num * n, I.data(), scope_vecs);

      const float **vecs = scope_vecs.Get();
      compute_vec(vecs);
    };
  } else {
    compute_dis = [&]() { compute_vec(nullptr); };
  }

  compute_dis();

#ifdef PERFORMANCE_TESTING
  condition->Perf("reorder");
#endif
  return 0;
}

}  // namespace gamma_gpu
}  // namespace tig_gamma
