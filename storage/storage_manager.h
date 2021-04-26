/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sstream>
#include <vector>
#include <tbb/concurrent_vector.h>

#include "async_writer.h"
#include "compress/compressor_zfp.h"
#include "compress/compressor_zstd.h"
#include "lru_cache.h"
#include "segment.h"
#include "vector_buffer_queue.h"

namespace tig_gamma {

struct StorageManagerOptions {
  int segment_size;
  int fixed_value_bytes;
  uint32_t seg_block_capacity;

  StorageManagerOptions() {
    segment_size = -1;
    fixed_value_bytes = -1;
    seg_block_capacity = 0;
  }

  StorageManagerOptions(const StorageManagerOptions &options) {
    segment_size = options.segment_size;
    fixed_value_bytes = options.fixed_value_bytes;
    seg_block_capacity = options.seg_block_capacity;
  }

  bool IsValid() {
    if (segment_size == -1 || fixed_value_bytes == -1 ||
        seg_block_capacity == 0) return false;
    return true;
  }

  std::string ToStr() {
    std::stringstream ss;
    ss << "{segment_size=" << segment_size
       << ", fixed_value_bytes=" << fixed_value_bytes
       << ", seg_block_capacity=" << seg_block_capacity << "}";
    return ss.str();
  }
};

class StorageManager {
 public:
  StorageManager(const std::string &root_path, BlockType block_type,
                 const StorageManagerOptions &options);
  ~StorageManager();
  int Init(int cache_size, std::string cache_name,
           int str_cache_size = 0, std::string str_cache_name = "");

  int Add(const uint8_t *value, int len);

  str_offset_t AddString(const char *value, int len, uint32_t &block_id,
                         uint32_t &in_block_pos);

  int Update(int id, uint8_t *value, int len);

  str_offset_t UpdateString(int id, const char *value, int len,
                            uint32_t &block_id, uint32_t &in_block_pos);

  // warning: vec can't be free
  int Get(long id, const uint8_t *&value);

  int GetString(long id, std::string &value, uint32_t blocck_id,
                uint32_t in_block_pos, str_len_t len);

  int GetHeaders(int start, int n, std::vector<const uint8_t *> &values,
                 std::vector<int> &lens);

  // currently it must call truncate after loading to set size of gamma db
  int Truncate(size_t size);

  int Size() { return size_; }

  int UseCompress(CompressType type, int d = -1, double rate = -1);

  bool AlterCacheSize(uint32_t cache_size, uint32_t str_cache_size);

  void GetCacheSize(uint32_t &cache_size, uint32_t &str_cache_size);

  void CountByteSize(uint64_t &base_size, uint64_t &str_size);

  int Sync();

 private:
  int Load();

  int Extend();

  std::string NextSegmentFilePath();

 private:
  std::string root_path_;
  StorageManagerOptions options_;
  size_t size_;                                // The total number of doc.
  tbb::concurrent_vector<Segment *> segments_;
  disk_io::AsyncWriter *disk_io_;
  BlockType block_type_;
  LRUCache<uint32_t, ReadFunParameter *> *cache_;
  LRUCache<uint32_t, ReadStrFunParameter *> *str_cache_;
  Compressor *compressor_;
};

}  // namespace tig_gamma
