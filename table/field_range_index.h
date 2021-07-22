/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef FIELD_RANGE_INDEX_H_
#define FIELD_RANGE_INDEX_H_

#include <map>
#include <string>
#include <vector>
#include <tbb/concurrent_queue.h>

#include "concurrentqueue/blockingconcurrentqueue.h"
#include "range_query_result.h"
#include "table.h"

#define    STR_MAX_INDEX_LEN    255

namespace tig_gamma {

enum class FilterOperator : uint8_t { And = 0, Or, Not };

typedef struct {
  int field;
  std::string lower_value;
  std::string upper_value;
  FilterOperator is_union;
} FilterInfo;

class ResourceToRecovery {
 public:
  explicit ResourceToRecovery(void *data, int after = 1) {
    deadline_ = std::chrono::system_clock::now() + std::chrono::seconds(after);
    data_ = data;
  }

  ~ResourceToRecovery() {
    free(data_);
    data_ = nullptr;
  }

  std::chrono::time_point<std::chrono::system_clock> Deadline() {
    return deadline_;
  }

  void *Data() { return data_; }

 private:
  std::chrono::time_point<std::chrono::system_clock> deadline_;
  void *data_;
};

class FieldOperate {
 public:
  typedef enum { ADD, DELETE } operate_type;
  explicit FieldOperate(operate_type type, int doc_id, int field_id)
      : type(type), doc_id(doc_id), field_id(field_id) {}

  operate_type type;
  int doc_id;
  int field_id;
  std::string value;
};

typedef moodycamel::BlockingConcurrentQueue<ResourceToRecovery *> ResourceQueue;
typedef tbb::concurrent_bounded_queue<FieldOperate *> FieldOperateQueue;

class FieldRangeIndex;
class MultiFieldsRangeIndex {
 public:
  MultiFieldsRangeIndex(std::string &path, table::Table *table);
  ~MultiFieldsRangeIndex();

  int Add(int docid, int field);

  int Delete(int docid, int field);

  int AddField(int field, enum DataType field_type);

  int Search(const std::vector<FilterInfo> &origin_filters,
             MultiRangeQueryResults *out);

  // for debug
  long MemorySize(long &dense, long &sparse);

 private:
  int Intersect(std::vector<RangeQueryResult> &results, int shortest_idx,
                RangeQueryResult *out);
  void ResourceRecoveryWorker();
  void FieldOperateWorker();

  int AddDoc(int docid, int field);

  int DeleteDoc(int docid, int field, std::string &key);
  std::vector<FieldRangeIndex *> fields_;
  table::Table *table_;
  std::string path_;
  bool b_running_;
  bool b_recovery_running_;
  bool b_operate_running_;
  ResourceQueue *resource_recovery_q_;
  FieldOperateQueue *field_operate_q_;
};

}  // namespace tig_gamma

#endif
