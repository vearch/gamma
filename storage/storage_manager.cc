/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "storage_manager.h"

#include "error_code.h"
#include "log.h"
#include "table_block.h"
#include "utils.h"
#include "vector_block.h"

namespace tig_gamma {

StorageManager::StorageManager(const std::string &root_path,
                               BlockType block_type,
                               const StorageManagerOptions &options)
    : root_path_(root_path), block_type_(block_type), options_(options) {
  size_ = 0;
  cache_ = nullptr;
  str_cache_ = nullptr;
  compressor_ = nullptr;
}

StorageManager::~StorageManager() {
  for (size_t i = 0; i < segments_.size(); i++) {
    CHECK_DELETE(segments_[i]);
  }
  delete disk_io_;
  disk_io_ = nullptr;
  CHECK_DELETE(str_cache_);
  CHECK_DELETE(cache_);
  CHECK_DELETE(compressor_);
}

std::string StorageManager::NextSegmentFilePath() {
  char buf[7];
  snprintf(buf, 7, "%06d", (int)segments_.size());
  std::string file_path = root_path_ + "/" + buf;
  return file_path;
}

int StorageManager::UseCompress(CompressType type, int d, double rate) {
  if (type == CompressType::Zfp) {
#ifdef WITH_ZFP
    if (d > 0) {
      compressor_ = new CompressorZFP(type);
      compressor_->Init(d);
    }
#endif
  }
  return (compressor_ ? 0 : -1);
}

bool StorageManager::AlterCacheSize(uint32_t cache_size,
                                    uint32_t str_cache_size) {
  if (cache_size > 0) {  // cache_size unit: M
    if (cache_ != nullptr) {
      cache_->AlterCacheSize((size_t)cache_size);
    } else {
      LOG(WARNING) << "Alter cache_ failure, cache_ is nullptr.";
    }
  }
  if (str_cache_size > 0) {
    if (str_cache_ != nullptr) {
      str_cache_->AlterCacheSize((size_t)str_cache_size);
    } else {
      LOG(WARNING) << "Alter str_cache_ failure, str_cache_ is nullptr.";
    }
  }
  return true;
}

void StorageManager::GetCacheSize(uint32_t &cache_size,
                                  uint32_t &str_cache_size) {
  if (cache_ != nullptr) {
    size_t max_size = cache_->GetMaxSize();
    cache_size = (uint32_t)(max_size * 64 / 1024);
  } else {
    cache_size = 0;
  }
  if (str_cache_ != nullptr) {
    size_t max_size = str_cache_->GetMaxSize();
    str_cache_size = (uint32_t)(max_size * 64 / 1024);
  } else {
    str_cache_size = 0;
  }
}

int StorageManager::Init(int cache_size, std::string cache_name,
                         int str_cache_size, std::string str_cache_name) {
  LOG(INFO) << "lrucache cache_size[" << cache_size
            << "M], string lrucache cache_size[" << str_cache_size << "M]";
  auto fun = &TableBlock::ReadBlock;
  if (block_type_ == BlockType::VectorBlockType) {
    fun = &VectorBlock::ReadBlock;
  }
  uint32_t per_block_size = ((64 * 1024) / options_.fixed_value_bytes) *
                            options_.fixed_value_bytes;  // block~=64k
  if (cache_size > 0) {
    cache_ = new LRUCache<uint32_t, ReadFunParameter *>(cache_name, cache_size,
                                                        per_block_size, fun);
    cache_->Init();
  }
  if (str_cache_size > 0) {
    str_cache_ = new LRUCache<uint32_t, ReadStrFunParameter *>(
        str_cache_name, str_cache_size, per_block_size,
        &StringBlock::ReadString);
    str_cache_->Init();
  }

  disk_io_ = new disk_io::AsyncWriter();
  if (!options_.IsValid()) {
    LOG(ERROR) << "invalid options=" << options_.ToStr();
    return PARAM_ERR;
  }
  if (utils::make_dir(root_path_.c_str())) {
    LOG(ERROR) << "mkdir error, path=" << root_path_;
    return IO_ERR;
  }

  Load();
  // init the first segment
  if (segments_.size() == 0 && Extend()) {
    return INTERNAL_ERR;
  }
  LOG(INFO) << "init gamma storage success! options=" << options_.ToStr()
            << ", segment num=" << segments_.size();
  return 0;
}

int StorageManager::Load() {
  // load existed segments
  while (utils::file_exist(NextSegmentFilePath())) {
    Segment *segment = new Segment(
        NextSegmentFilePath(), segments_.size(), options_.segment_size,
        options_.fixed_value_bytes, options_.seg_block_capacity, disk_io_,
        (void *)cache_, (void *)str_cache_);
    int ret = segment->Load(block_type_, compressor_);
    if (ret < 0) {
      LOG(ERROR) << "extend file segment error, ret=" << ret;
      return ret;
    }
    size_ += ret;
    segments_.push_back(segment);
  }

  LOG(INFO) << "load gamma storage success! options=" << options_.ToStr()
            << ", segment num=" << segments_.size();
  return size_;
}

int StorageManager::Extend() {
  uint32_t seg_id = (uint32_t)segments_.size();
  Segment *segment =
      new Segment(NextSegmentFilePath(), seg_id, options_.segment_size,
                  options_.fixed_value_bytes, options_.seg_block_capacity,
                  disk_io_, (void *)cache_, (void *)str_cache_);
  int ret = segment->Init(block_type_, compressor_);
  if (ret) {
    LOG(ERROR) << "extend file segment error, ret=" << ret;
    return ret;
  }
  segments_.push_back(segment);
  return 0;
}

int StorageManager::Add(const uint8_t *value, int len) {
  if (len != options_.fixed_value_bytes) {
    LOG(ERROR) << "Add len error [" << len << "] != options_.fixed_value_bytes["
               << options_.fixed_value_bytes << "]";
    return PARAM_ERR;
  }

  Segment *segment = segments_.back();
  int ret = segment->Add(value, len);
  if (ret) {
    LOG(ERROR) << "segment add error [" << ret << "]";
    return ret;
  }
  ++size_;

  if (segment->IsFull() && Extend()) {
    LOG(ERROR) << "extend error";
    return INTERNAL_ERR;
  }
  return 0;
}

str_offset_t StorageManager::AddString(const char *value, int len,
                                       uint32_t &block_id,
                                       uint32_t &in_block_pos) {
  Segment *segment = segments_.back();
  str_offset_t ret = segment->AddString(value, len, block_id, in_block_pos);
  return ret;
}

int StorageManager::GetHeaders(int start, int n,
                               std::vector<const uint8_t *> &vecs,
                               std::vector<int> &lens) {
  if ((size_t)start + n > size_) {
    LOG(ERROR) << "start [" << start << "] + n [" << n << "] > size_ [" << size_
               << "]";
    return PARAM_ERR;
  }
  while (n) {
    int offset = start % options_.segment_size;
    int len = options_.segment_size - offset;
    if (len > n) len = n;
    lens.push_back(len);
    Segment *segment = segments_[start / options_.segment_size];
    uint8_t *value = new uint8_t[len * options_.fixed_value_bytes];
    segment->GetValues(value, offset, len);
    // std::stringstream ss;
    // for (int i = 0; i < 100; ++i) {
    //   float a;
    //   memcpy(&a, value + i * 4, 4);
    //   ss << a << " ";
    // }
    // std::string aa = ss.str();
    vecs.push_back(value);
    start += len;
    n -= len;
  }
  return 0;
}

int StorageManager::Update(int id, uint8_t *v, int len) {
  if ((size_t)id >= size_ || id < 0 || len != options_.fixed_value_bytes) {
    LOG(ERROR) << "id [" << id << "] size_ [" << size_ << "]";
    return PARAM_ERR;
  }
  return segments_[id / options_.segment_size]->Update(
      id % options_.segment_size, v, len);
}

str_offset_t StorageManager::UpdateString(int id, const char *value, int len,
                                          uint32_t &block_id,
                                          uint32_t &in_block_pos) {
  if ((size_t)id >= size_ || id < 0) {
    LOG(ERROR) << "id [" << id << "] size_ [" << size_ << "]";
    return PARAM_ERR;
  }
  int seg_id = id / options_.segment_size;
  int count = 0;
  while (seg_id >= segments_.size()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    LOG(INFO) << "Get(),seg_id:" << seg_id
              << " >= segments_.size():" << segments_.size();
    ++count;
    if (count > 10) {
      LOG(ERROR) << "Because the wait timeout, StorageManager["
                 << str_cache_->GetName() << "] UpdateString(" << id
                 << ") failed.";
      return -1;
    }
  }
  Segment *segment = segments_[seg_id];
  str_offset_t ret = segment->AddString(value, len, block_id, in_block_pos);
  return ret;
}

int StorageManager::Get(long id, const uint8_t *&value) {
  if ((size_t)id >= size_ || id < 0) {
    LOG(WARNING) << "id [" << id << "] size_ [" << size_ << "]";
    return PARAM_ERR;
  }

  int seg_id = id / options_.segment_size;
  int count = 0;
  while (seg_id >= segments_.size()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ++count;
    if (count % 20 == 0) {
      LOG(WARNING) << "Waited " << count * 20 << "ms, StorageManager["
                   << cache_->GetName() << "] Get(" << id << "), seg_id["
                   << seg_id << "] >= segments_.size()[" << segments_.size() << "].";
    }
  }
  Segment *segment = segments_[seg_id];

  uint8_t *value2 = new uint8_t[options_.fixed_value_bytes];
  int ret = segment->GetValue(value2, id % options_.segment_size);
  value = value2;
  return ret;
}

int StorageManager::GetString(long id, std::string &value, uint32_t block_id,
                              uint32_t in_block_pos, str_len_t len) {
  if ((size_t)id >= size_ || id < 0) {
    LOG(ERROR) << "id [" << id << "] size_ [" << size_ << "]";
    return PARAM_ERR;
  }
  // TODO wait while seg_id >= segments_.size()
  int seg_id = id / options_.segment_size;
  int count = 0;
  while (seg_id >= segments_.size()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ++count;
    if (count % 20 == 0) {
      LOG(ERROR) << "Waited " << count * 20 << "ms, StorageManager["
                 << cache_->GetName() << "] GetString(" << id << ") failed. seg_id["
                 << seg_id << "] >= segments_.size()[" << segments_.size() << "].";
    }
  }

  value = segments_[seg_id]->GetString(block_id, in_block_pos, len);
  return 0;
}

int StorageManager::Truncate(size_t size) {
  size_t seg_num = size / options_.segment_size;
  size_t offset = size % options_.segment_size;
  if (offset > 0) ++seg_num;
  if (seg_num > segments_.size()) {
    LOG(ERROR) << "gamma storage only has " << segments_.size()
               << " segments, but expect " << seg_num
               << ", trucate size=" << size;
    return PARAM_ERR;
  }

  if (segments_.size() > seg_num) {
    for (size_t i = seg_num; i < segments_.size(); ++i) {
      delete segments_[i];
      segments_[i] = nullptr;
    }
  }

  segments_.resize(seg_num);
  if (offset > 0) {
    segments_.back()->SetBaseSize((uint32_t)offset);
  }
  size_ = size;

  if (seg_num == 0 && Extend()) {
    return INTERNAL_ERR;
  }
  LOG(INFO) << "gamma storage truncate to size=" << size
            << ", current segment num=" << segments_.size()
            << ", last offset=" << offset;
  return 0;
}

void StorageManager::CountByteSize(uint64_t &base_size, uint64_t &str_size) {
  base_size = 0;
  str_size = 0;
  int n = segments_.size();
  for (int i = 0; i < n; ++i) {
    Segment *seg = segments_[i];
    uint32_t base_off = seg->BaseOffset();
    str_offset_t str_off = seg->StrOffset();
    base_size += base_off;
    str_size += str_off;
  }
}

int StorageManager::Sync() { return disk_io_->Sync(); }

}  // namespace tig_gamma
