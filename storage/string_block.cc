/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "string_block.h"

#include <unistd.h>
// #include <concurrent_vector.h>

namespace tig_gamma {

const static int MAX_STR_BLOCK_SIZE = 102400;

StringBlock::StringBlock(int fd, int per_block_size, int length,
                         uint32_t header_size, uint32_t seg_id,
                         uint32_t seg_block_capacity)
    : Block(fd, per_block_size, length, header_size, seg_id,
            seg_block_capacity) {}

StringBlock::~StringBlock() {
  if (block_pos_fp_ != nullptr) {
    fclose(block_pos_fp_);
    block_pos_fp_ = nullptr;
  }
}

void StringBlock::InitStrBlock(void *lru) {
  lru_cache_ =
      (LRUCache<uint32_t, std::vector<uint8_t>, ReadStrFunParameter *> *)lru;
}

int StringBlock::LoadIndex(const std::string &file_path) {
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

int StringBlock::CloseBlockPosFile() {
  if (block_pos_fp_ != nullptr) {
    fclose(block_pos_fp_);
    block_pos_fp_ = nullptr;
    return 0;
  }
  return -1;
}

int StringBlock::WriteContent(const uint8_t *data, int len, uint32_t offset,
                              disk_io::AsyncWriter *disk_io) {
  return 0;
}

int StringBlock::ReadContent(uint8_t *value, uint32_t len, uint32_t offset) {
  return 0;
}

int StringBlock::WriteString(const char *data, str_len_t len,
                             str_offset_t offset, uint32_t &block_id,
                             uint32_t &in_block_pos) {
  if (block_pos_.size() == 0) {
    AddBlockPos(0);
  }
  uint32_t cur_pos = block_pos_.back();
  in_block_pos = offset - cur_pos;
  block_id = block_pos_.size() - 1;
  if (in_block_pos + len > MAX_STR_BLOCK_SIZE) {
    pwrite(fd_, data, len, offset);
    AddBlockPos(offset + len);
  } else {
    pwrite(fd_, data, len, offset);
  }
  return 0;
}

int StringBlock::Read(uint32_t block_id, uint32_t in_block_pos, str_len_t len,
                      std::string &str_out) {
  // uint32_t off = block_pos_[block_id] + in_block_pos;
  // std::vector<char> str(len);
  // pread(fd_, str.data(), str.size(), off);
  // str_out = std::move(std::string(str.data(), len));

  char *str = new char[len];
  uint32_t block_pos = block_pos_[block_id];

  uint32_t block_pos_size = block_pos_.size();
  // TODO needn't read last block's disk if it is not in last segment
  if (block_id + 1 >= block_pos_size) {
    pread(fd_, str, len, block_pos + in_block_pos);
  } else {
    std::shared_ptr<std::vector<uint8_t>> block;
    uint32_t cache_bid = GetCacheBlockId(block_id);
    bool res = lru_cache_->Get(cache_bid, block);
    if (not res) {
      ReadStrFunParameter parameter;
      parameter.str_block = this;
      parameter.block_id = block_id;
      parameter.in_block_pos = in_block_pos;
      parameter.fd = fd_;  // TODO remove
      res = lru_cache_->SetOrGet(cache_bid, block, &parameter);
    }

    if (not res) {
      LOG(ERROR) << "Read block fails from disk_file, block_id[" << block_id
                 << "]";
      delete[] str;
      return -1;
    }
    memcpy(str, block->data() + in_block_pos, len);
  }

  str_out = std::string(str, len);
  delete[] str;
  return 0;
}

bool StringBlock::ReadString(uint32_t key,
                             std::shared_ptr<std::vector<uint8_t>> &block,
                             ReadStrFunParameter *param) {
  StringBlock *str_block = reinterpret_cast<StringBlock *>(param->str_block);
  uint32_t len = str_block->block_pos_[param->block_id + 1] -
                 str_block->block_pos_[param->block_id];
  block = std::make_shared<std::vector<uint8_t>>(len);
  uint32_t cur_pos = str_block->block_pos_[param->block_id];  // TODO check size
  pread(param->fd, block->data(), len, cur_pos);
  return true;
}

int StringBlock::AddBlockPos(uint32_t block_pos) {
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

}  // namespace tig_gamma