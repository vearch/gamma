/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "block.h"

#include <unistd.h>

namespace tig_gamma {

Block::Block(int fd, int per_block_size, int length, uint32_t header_size)
    : fd_(fd),
      per_block_size_(per_block_size),
      item_length_(length),
      header_size_(header_size) {
  compressor_ = nullptr;
  size_ = 0;
  block_pos_fp_ = nullptr;
}

Block::~Block() {
  lru_cache_ = nullptr;
  compressor_ = nullptr;
  if (block_pos_fp_ != nullptr) {
    fclose(block_pos_fp_);
    block_pos_fp_ = nullptr;
  }
}

void Block::Init(void *lru, Compressor *compressor) {
  lru_cache_ =
      (LRUCache<uint64_t, std::vector<uint8_t>, ReadFunParameter *> *)lru;
  compressor_ = compressor;
  InitSubclass();
}

int Block::LoadIndex(const std::string &file_path) {
  FILE *file = fopen(file_path.c_str(), "rb");
  if (file != nullptr) {
    size_t read_num = 0;
    do {
      uint32_t pos;
      read_num = fread(&pos, sizeof(pos), 1, file);
      if (read_num == 0) {
        break;
      }
      block_pos_.push_back(pos);
    } while (read_num != 0);

    fclose(file);
  }

  block_pos_fp_ = fopen(file_path.c_str(), "ab+");
  if (block_pos_fp_ == nullptr) {
    LOG(ERROR) << "open block pos file error, path=" << file_path;
    return -1;
  }
  block_pos_file_path_ = file_path;
  return 0;
}

int Block::AddBlockPos(uint32_t block_pos) {
  block_pos_.push_back(block_pos);
  bool is_close = false;
  if (block_pos_fp_ == nullptr) {
    block_pos_fp_ = fopen(block_pos_file_path_.c_str(), "ab+");
    if (block_pos_fp_ == nullptr) {
      LOG(ERROR) << "open block pos file error, path="
                 << block_pos_file_path_;
      return -1;
    }
    is_close = true;
  }
  fwrite(&block_pos, sizeof(block_pos), 1, block_pos_fp_);
  fflush(block_pos_fp_);
  if (is_close) {
    CloseBlockPosFile();
  }
  return 0;
}

int Block::Write(const uint8_t *value, int n_bytes, uint32_t start,
                 disk_io::AsyncWriter *disk_io) {
  if (size_ / per_block_size_ >= block_pos_.size()) {
    AddBlockPos(start);
    // compress prev
  }
  size_ += n_bytes;
  WriteContent(value, n_bytes, start, disk_io);
  return 0;
}

static uint32_t WritenSize(int fd) {
  uint32_t size;
  pread(fd, &size, sizeof(size), sizeof(uint8_t) + sizeof(uint32_t));
  return size;
}

int Block::Read(uint8_t *value, uint32_t n_bytes, uint32_t start) {
  int read_num = 0;
  while (n_bytes) {
    int len = n_bytes;
    if (len > per_block_size_) len = per_block_size_;

    uint32_t block_id = start / per_block_size_;
    uint32_t block_pos = block_pos_[block_id];
    uint32_t block_offset = start % per_block_size_;

    if (len > per_block_size_ - block_offset)
      len = per_block_size_ - block_offset;

    uint32_t cur_size = WritenSize(fd_);
    uint32_t b = cur_size * item_length_ / per_block_size_;
    // TODO needn't read last block's disk if it is not in last segment
    if (b <= block_id) {
      pread(fd_, value + read_num, len,
            block_pos + header_size_ + block_offset);
    } else {
      std::shared_ptr<std::vector<uint8_t>> block;
      uint64_t uni_block_id = block_id;
      uni_block_id = uni_block_id << 32;
      uni_block_id |= fd_;
      bool res = lru_cache_->Get(uni_block_id, block);
      if (not res) {
        ReadFunParameter parameter;
        parameter.len = per_block_size_;
        parameter.offset = block_pos;
        parameter.fd = fd_;  // TODO remove
        parameter.cmprs = (void*)compressor_;
        GetReadFunParameter(parameter);
        res = lru_cache_->SetOrGet(uni_block_id, block, &parameter);
      }

      if (not res) {
        LOG(ERROR) << "Read block fails from disk_file, block_id[" << block_id
                   << "]";
        return -1;
      }
      memcpy(value + read_num, block->data() + block_offset, len);
    }

    start += len;
    read_num += len;
    n_bytes -= len;
  }
  return 0;
}

int Block::Update(const uint8_t *data, int n_bytes, uint32_t offset) {
  int res = SubclassUpdate(data, n_bytes, offset);
  if (res != 0) return res;

  while (n_bytes) {
    int len = n_bytes;
    if (len > per_block_size_) len = per_block_size_;

    uint32_t block_id = offset / per_block_size_;
    uint32_t block_offset = offset % per_block_size_;

    if (len > per_block_size_ - block_offset)
      len = per_block_size_ - block_offset;

    uint64_t uni_block_id = block_id;
    uni_block_id = uni_block_id << 32;
    uni_block_id |= fd_;
    lru_cache_->Evict(uni_block_id);

    offset += len;
    n_bytes -= len;
  }
  return res;
}


int Block::CloseBlockPosFile() {
  if (block_pos_fp_ != nullptr) {
    fclose(block_pos_fp_);
    block_pos_fp_ = nullptr;
    return 0;
  }
  return -1;
}

}  // namespace tig_gamma