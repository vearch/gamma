/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "segment.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error_code.h"
#include "log.h"
#include "table_block.h"
#include "vector_block.h"
#include "thread_util.h"
#include "utils.h"

namespace tig_gamma {

namespace {

inline size_t CapacityOff() { return sizeof(uint8_t); }

inline size_t SizeOff() {
  uint32_t capacity;
  return CapacityOff() + sizeof(capacity);
}

inline size_t StrCapacityOff() {
  uint32_t size;
  return SizeOff() + sizeof(size);
}

inline size_t StrSizeOff() {
  uint64_t str_capacity;
  return StrCapacityOff() + sizeof(str_capacity);
}

inline size_t StrBlocksSizeOff() {
  uint64_t str_capacity;
  return StrSizeOff() + sizeof(str_capacity);
}

inline size_t StrCompressedOff() {
  str_offset_t str_size;
  return StrBlocksSizeOff() + sizeof(str_size);
}

inline size_t BCompressedOff() {
  str_offset_t str_compressed_size;
  return StrCompressedOff() + sizeof(str_compressed_size);
}

}  // namespace

Segment::Segment(const std::string &file_path, int max_size, int vec_byte_size,
                 disk_io::AsyncWriter *disk_io, void *table_cache,
                 void *str_cache)
    : file_path_(file_path),
      max_size_(max_size),
      item_length_(vec_byte_size),
      disk_io_(disk_io),
      cache_(table_cache),
      str_cache_(str_cache) {
  base_fd_ = -1;
  str_fd_ = -1;

  cur_size_ = 0;
  capacity_ = 0;
  version_ = 0;
  uint32_t capacity;
  uint32_t size;
  uint64_t str_capacity;
  str_offset_t str_size;
  uint32_t str_blocks_size;
  str_offset_t str_compressed_size;
  uint8_t b_compressed;

  seg_header_size_ = sizeof(version_) + sizeof(capacity) + sizeof(size) +
                     sizeof(str_capacity) + sizeof(str_size) +
                     sizeof(str_blocks_size) + sizeof(str_compressed_size) +
                     sizeof(b_compressed);
  mapped_byte_size_ = (size_t)max_size * item_length_ + seg_header_size_;

  per_block_size_ = ((64 * 1024) / item_length_) * item_length_; // block~=64k
  buffered_size_ = 0;
}

Segment::~Segment() {
  if (base_fd_ != -1) {
    close(base_fd_);
    base_fd_ = -1;
  }

  if (str_fd_ != -1) {
    close(str_fd_);
    str_fd_ = -1;
  }

  if (blocks_ != nullptr) {
    delete blocks_;
    blocks_ = nullptr;
  }

  if (str_blocks_ != nullptr) {
    delete str_blocks_;
    str_blocks_ = nullptr;
  }
}

uint8_t Segment::Version() {
  uint8_t version = 0;
  pread(base_fd_, &version, sizeof(version), 0);
  return version;
}

void Segment::SetVersion(uint8_t version) {
  pwrite(base_fd_, &version, sizeof(version), 0);
}

uint32_t Segment::BufferedSize() { return buffered_size_; }

void Segment::PersistentedSize() {
  uint32_t capacity;
  uint32_t size = 0; 
  pread(base_fd_, &size, sizeof(size), sizeof(version_) + sizeof(capacity));
  cur_size_ = size;
}

void Segment::SetSize(uint32_t size) {
  uint32_t capacity;
  pwrite(base_fd_, &size, sizeof(size), sizeof(version_) + sizeof(capacity));
}

uint64_t Segment::StrCapacity() {
  uint64_t str_capacity;
  pread(base_fd_, &str_capacity, sizeof(str_capacity), StrCapacityOff());
  return str_capacity;
}

void Segment::SetStrCapacity(uint64_t str_capacity) {
  pwrite(base_fd_, &str_capacity, sizeof(str_capacity), StrCapacityOff());
}

uint32_t Segment::StrBlocksSize() {
  uint32_t str_blocks_size;
  pread(base_fd_, &str_blocks_size, sizeof(str_blocks_size),
        StrBlocksSizeOff());
  return str_blocks_size;
}

void Segment::SetStrBlocksSize(uint32_t str_blocks_size) {
  pwrite(base_fd_, &str_blocks_size, sizeof(str_blocks_size),
         StrBlocksSizeOff());
}

str_offset_t Segment::StrSize() {
  str_offset_t str_size;
  pread(base_fd_, &str_size, sizeof(str_size), StrSizeOff());
  return str_size;
}

void Segment::SetStrSize(str_offset_t str_size) {
  pwrite(base_fd_, &str_size, sizeof(str_size), StrSizeOff());
}

uint8_t Segment::BCompressed() {
  uint8_t b_compressed;
  pread(base_fd_, &b_compressed, sizeof(b_compressed), BCompressedOff());
  return b_compressed;
}

void Segment::SetCompressed(uint8_t compressed) {
  pwrite(base_fd_, &compressed, sizeof(compressed), BCompressedOff());
}

str_offset_t Segment::StrCompressedSize() {
  str_offset_t str_compressed_size;
  pread(base_fd_, &str_compressed_size, sizeof(str_compressed_size),
        StrCompressedOff());
  return str_compressed_size;
}

void Segment::SetStrCompressedSize(str_offset_t str_compressed_size) {
  pwrite(base_fd_, &str_compressed_size, sizeof(str_compressed_size),
         StrCompressedOff());
}

int Segment::Init(BlockType block_type, Compressor *compressor) {
  OpenFile();
  if (ftruncate(base_fd_, seg_header_size_ + item_length_ * max_size_)) {
    close(base_fd_);
    LOG(ERROR) << "truncate file error:" << strerror(errno);
    return IO_ERR;
  }

  uint64_t str_capacity = seg_header_size_ + max_size_ * 4;
  SetStrCapacity(str_capacity);
  SetStrSize(0);
  int ret = ftruncate(str_fd_, StrCapacity());
  if (ret != 0) {
    return -1;
  }
  InitBlock(block_type, compressor);

  return 0;
}

int Segment::OpenFile() {
  base_fd_ = open(file_path_.c_str(), O_RDWR | O_CREAT, 0666);
  if (-1 == base_fd_) {
    LOG(ERROR) << "open vector file error, path=" << file_path_;
    return IO_ERR;
  }

  str_fd_ = open((file_path_ + "_str").c_str(), O_RDWR | O_CREAT, 0666);
  if (-1 == str_fd_) {
    LOG(ERROR) << "open vector file error, path=" << (file_path_ + "_str");
    return -1;
  }
  return 0;
}

int Segment::InitBlock(BlockType block_type, Compressor *compressor) {
  switch (block_type)
  {
  case BlockType::TableBlockType:
    blocks_ =
      new TableBlock(base_fd_, per_block_size_, item_length_, seg_header_size_);
    break;
  case BlockType::VectorBlockType:
    blocks_ =
      new VectorBlock(base_fd_, per_block_size_, item_length_, seg_header_size_);
    break;
  default:
    LOG(ERROR) << "BlockType is error";
    break;
  }

  blocks_->Init(cache_, compressor);
  blocks_->LoadIndex(file_path_ + ".idx");

  str_blocks_ =
      new StringBlock(str_fd_, 1024 * 1024, item_length_, seg_header_size_);
  str_blocks_->LoadIndex(file_path_ + "_str.idx");
  str_blocks_->InitStrBlock(str_cache_);
  if (BufferedSize() == max_size_) {
    blocks_->CloseBlockPosFile();
    str_blocks_->CloseBlockPosFile();
  }
  return 0;
}

// TODO: Load compressor
int Segment::Load(BlockType block_type, Compressor *compressor) {
  OpenFile();
  InitBlock(block_type, compressor);

  uint64_t str_capacity = StrCapacity();
  PersistentedSize();
  return cur_size_;
}

int Segment::Add(const uint8_t *data, int len) {
  size_t offset = (size_t)buffered_size_ * item_length_;
  blocks_->Write(data, len, offset, disk_io_);
  ++buffered_size_;
  return 0;
}

str_offset_t Segment::AddString(const char *str, int len, uint32_t &block_id,
                                uint32_t &in_block_pos) {
  str_offset_t str_size = StrSize();
  uint64_t str_capacity = StrCapacity();
  if (str_size + len >= str_capacity) {
    uint64_t extend_capacity = str_capacity << 1;
    while (str_size + len >= extend_capacity) {
      extend_capacity = extend_capacity << 1;
    }

    int ret = 0;
    SetStrCapacity(extend_capacity);

    ret = ftruncate(str_fd_, StrCapacity());
    if (ret != 0) {
      return -1;
    }
  }

  str_blocks_->WriteString(str, len, str_size, block_id, in_block_pos);

  SetStrSize(str_size + len);
  return str_size;
}

int Segment::GetValue(uint8_t *value, int id) {
  return GetValues(value, id, 1);
}

int Segment::GetValues(uint8_t *value, int id, int n) {
  int start = id * item_length_;
  int n_bytes = n * item_length_;
  // TODO read from buffer queue
  while (id >= (int)cur_size_) {
    PersistentedSize();
    if (id < (int)cur_size_) break;
    LOG(INFO) << "Data not brushed disk, wait 10ms.";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  blocks_->Read(value, n_bytes, start);
  return 0;
}

std::string Segment::GetString(uint32_t block_id, uint32_t in_block_pos,
                               str_len_t len) {
  std::string str;
  str_blocks_->Read(block_id, in_block_pos, len, str);
  return str;
}

bool Segment::IsFull() {
  if (BufferedSize() == max_size_) {
    blocks_->CloseBlockPosFile();
    str_blocks_->CloseBlockPosFile();
    return true;
  } else {
    return false;
  }
}

int Segment::Update(int id, uint8_t *data, int len) {
  size_t offset = (size_t)id * item_length_;
  blocks_->Update(data, len, offset);
  return 0;
}

}  // namespace tig_gamma
