/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef WITH_ROCKSDB

#include "rocksdb_raw_vector.h"
#include <stdio.h>
#include "log.h"
#include "rocksdb/table.h"
#include "utils.h"

using namespace std;
using namespace rocksdb;

namespace tig_gamma {

template <typename DataType>
RocksDBRawVector<DataType>::RocksDBRawVector(const std::string &name, int dimension,
                                   int max_vector_size,
                                   const std::string &root_path,
                                   const StoreParams &store_params)
    : RawVector<DataType>(name, dimension, max_vector_size, root_path) {
  this->root_path_ = root_path;
  db_ = nullptr;
  store_params_ = new StoreParams(store_params);
}

template <typename DataType>
RocksDBRawVector<DataType>::~RocksDBRawVector() {
  if (db_) {
    delete db_;
  }
  if (store_params_) delete store_params_;
}

template <typename DataType>
int RocksDBRawVector<DataType>::InitStore() {
  block_cache_size_ = store_params_->cache_size_;

  std::shared_ptr<Cache> cache = NewLRUCache(block_cache_size_);
  // BlockBasedTableOptions table_options_;
  table_options_.block_cache = cache;
  Options options;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options_));

  options.IncreaseParallelism();
  // options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;

  string db_path = this->root_path_ + "/" + this->vector_name_;
  if (!utils::isFolderExist(db_path.c_str())) {
    mkdir(db_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  // open DB
  Status s = DB::Open(options, db_path, &db_);
  if (!s.ok()) {
    LOG(ERROR) << "open rocks db error: " << s.ToString();
    return -1;
  }
  LOG(INFO) << "rocks raw vector init success! name=" << this->vector_name_
            << ", block cache size=" << block_cache_size_ << "Bytes";

  return 0;
}

template <typename DataType>
int RocksDBRawVector<DataType>::GetVector(long vid, const DataType *&vec,
                                bool &deletable) const {
  if (vid >= this->ntotal_ || vid < 0) {
    return 1;
  }
  string key, value;
  ToRowKey((int)vid, key);
  Status s = db_->Get(ReadOptions(), Slice(key), &value);
  if (!s.ok()) {
    LOG(ERROR) << "rocksdb get error:" << s.ToString() << ", key=" << key;
    return 2;
  }
  DataType *vector = new DataType[this->dimension_];
  assert((size_t)this->vector_byte_size_ == value.size());
  memcpy((void *)vector, value.data(), this->vector_byte_size_);
  vec = vector;
  deletable = true;
  return 0;
}

template <typename DataType>
int RocksDBRawVector<DataType>::AddToStore(DataType *v, int len) {
  return UpdateToStore(this->ntotal_, v, len);
}

template <typename DataType>
size_t RocksDBRawVector<DataType>::GetStoreMemUsage() {
  size_t cache_mem = table_options_.block_cache->GetUsage();
  std::string index_mem;
  db_->GetProperty("rocksdb.estimate-table-readers-mem", &index_mem);
  std::string memtable_mem;
  db_->GetProperty("rocksdb.cur-size-all-mem-tables", &memtable_mem);
  size_t pin_mem = table_options_.block_cache->GetPinnedUsage();
  LOG(INFO) << "rocksdb mem usage: block cache=" << cache_mem
            << ", index and filter=" << index_mem
            << ", memtable=" << memtable_mem
            << ", iterators pinned=" << pin_mem;
  return 0;
}

template <typename DataType>
int RocksDBRawVector<DataType>::UpdateToStore(int vid, DataType *v, int len) {
  if (v == nullptr || len != this->dimension_) return -1;
  string key;
  ToRowKey(vid, key);
  Status s =
      db_->Put(WriteOptions(), Slice(key), Slice((char *)v, this->vector_byte_size_));
  if (!s.ok()) {
    LOG(ERROR) << "rocksdb update error:" << s.ToString() << ", key=" << key;
    return -2;
  }
  return 0;
}

template <typename DataType>
int RocksDBRawVector<DataType>::GetVectorHeader(int start, int end, ScopeVector<DataType> &vec) {
  if (start < 0 || start >= this->ntotal_ || start >= end) {
    return 1;
  }

  rocksdb::Iterator *it = db_->NewIterator(rocksdb::ReadOptions());
  string start_key, end_key;
  ToRowKey(start, start_key);
  ToRowKey(end, end_key);
  it->Seek(Slice(start_key));
  int num = end - start;
  DataType *vectors = new DataType[(uint64_t)this->dimension_ * num];
  for (int c = 0; c < num; c++, it->Next()) {
    if (!it->Valid()) {
      LOG(ERROR) << "rocksdb iterator error, count=" << c;
      delete it;
      return 2;
    }
    Slice value = it->value();
    assert(value.size_ == (size_t)this->vector_byte_size_);
    memcpy((void *)(vectors + (uint64_t)c * this->dimension_), value.data_,
           this->vector_byte_size_);
#ifdef DEBUG
    string expect_key;
    ToRowKey(c + start, expect_key);
    string key = it->key().ToString();
    if (key != expect_key) {
      LOG(ERROR) << "vid=" << c + start << ", invalid key=" << key
                 << ", expect=" << expect_key;
    }
#endif
  }
  delete it;
  vec.Set(vectors);
  return 0;
}

template <typename DataType>
void RocksDBRawVector<DataType>::ToRowKey(int vid, string &key) const {
  char data[11];
  snprintf(data, 11, "%010d", vid);
  key.assign(data, 10);
}

template class RocksDBRawVector<float>;
template class RocksDBRawVector<uint8_t>;
}  // namespace tig_gamma

#endif  // WITH_ROCKSDB
