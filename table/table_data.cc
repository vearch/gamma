/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "table_data.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <zstd.h>

#include <new>

#include "log.h"
#include "thread_util.h"
#include "utils.h"

#define MAX_MMAP_SIZE (1 << 30)

namespace tig_gamma {
namespace table {

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

inline size_t StrCompressedOff() {
  str_offset_t str_size;
  return StrSizeOff() + sizeof(str_size);
}

inline size_t BCompressedOff() {
  str_offset_t str_compressed_size;
  return StrCompressedOff() + sizeof(str_compressed_size);
}

}  // namespace

TableData::TableData(int item_length) {
  base_ = nullptr;
  base_str_ = nullptr;
  mmap_buf_ = nullptr;
  mmap_str_buf_ = nullptr;
  b_running_ = false;
  item_length_ = item_length;
  capacity_ = 0;
  version_ = 0;
  seg_header_backup_ = 20;
  uint32_t capacity;
  uint32_t size;
  uint64_t str_capacity;
  str_offset_t str_size;
  str_offset_t str_compressed_size;
  uint8_t b_compressed;
  mode_ = TABLE_LOAD_MODE::MODE_MEMORY_DISK;
  seg_header_size_ = sizeof(version_) + sizeof(capacity) + sizeof(size) +
                     sizeof(str_capacity) + sizeof(str_size) +
                     sizeof(str_compressed_size) + sizeof(b_compressed) +
                     seg_header_backup_;
}

TableData::~TableData() {
  if (base_) {
    delete[] base_;
    base_ = nullptr;
  }
  if (base_str_) {
    delete[] base_str_;
    base_str_ = nullptr;
  }
  munmap(mmap_buf_, MAX_MMAP_SIZE);
  munmap(mmap_str_buf_, MAX_MMAP_SIZE);
  pthread_rwlock_destroy(&shared_mutex_);
}

uint8_t TableData::Version() {
  uint8_t version = 0;
  memcpy(&version, base_, sizeof(version));
  return version;
}

void TableData::SetVersion(uint8_t version) {
  memcpy(base_, &version, sizeof(version));
  memcpy(mmap_buf_, &version, sizeof(version));
}

uint32_t TableData::Size() {
  uint32_t capacity;
  uint32_t size;
  memcpy(&size, base_ + sizeof(version_) + sizeof(capacity), sizeof(size));
  return size;
}

void TableData::SetSize(uint32_t size) {
  uint32_t capacity;
  memcpy(base_ + sizeof(version_) + sizeof(capacity), &size, sizeof(size));
  memcpy(mmap_buf_ + sizeof(version_) + sizeof(capacity), &size, sizeof(size));
}

uint64_t TableData::StrCapacity() {
  uint64_t str_capacity;
  memcpy(&str_capacity, base_ + StrCapacityOff(), sizeof(str_capacity));
  return str_capacity;
}

void TableData::SetStrCapacity(uint64_t str_capacity) {
  memcpy(base_ + StrCapacityOff(), &str_capacity, sizeof(str_capacity));
  memcpy(mmap_buf_ + StrCapacityOff(), &str_capacity, sizeof(str_capacity));
}

str_offset_t TableData::StrSize() {
  str_offset_t str_size;
  memcpy(&str_size, base_ + StrSizeOff(), sizeof(str_size));
  return str_size;
}

void TableData::SetStrSize(str_offset_t str_size) {
  memcpy(base_ + StrSizeOff(), &str_size, sizeof(str_size));
  memcpy(mmap_buf_ + StrSizeOff(), &str_size, sizeof(str_size));
}

uint8_t TableData::BCompressed() {
  uint8_t b_compressed;
  memcpy(&b_compressed, base_ + BCompressedOff(), sizeof(b_compressed));
  return b_compressed;
}

void TableData::SetCompressed(uint8_t compressed) {
  memcpy(base_ + BCompressedOff(), &compressed, sizeof(compressed));
  memcpy(mmap_buf_ + BCompressedOff(), &compressed, sizeof(compressed));
}

str_offset_t TableData::StrCompressedSize() {
  str_offset_t str_compressed_size;
  memcpy(&str_compressed_size, base_ + StrCompressedOff(),
         sizeof(str_compressed_size));
  return str_compressed_size;
}

void TableData::SetStrCompressedSize(str_offset_t str_compressed_size) {
  memcpy(base_ + StrCompressedOff(), &str_compressed_size,
         sizeof(str_compressed_size));
  memcpy(mmap_buf_ + StrCompressedOff(), &str_compressed_size,
         sizeof(str_compressed_size));
}

int TableData::Sync() {
  if (not b_running_) return 0;

  WriteThreadLock write_lock(shared_mutex_);
  int ret = 0;
  ret = msync(mmap_buf_, MAX_MMAP_SIZE, MS_SYNC);
  if (ret != 0) {
    LOG(ERROR) << "id [" << id_ << "] sync failed!";
    return ret;
  }
  ret = msync(mmap_str_buf_, MAX_MMAP_SIZE, MS_SYNC);
  if (ret != 0) {
    LOG(ERROR) << "id [" << id_ << "] sync failed!";
    return ret;
  }
  return ret;
}

int TableData::Init(int id, const std::string &path, uint8_t string_field_num) {
  base_ = new (
      std::nothrow) char[seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT];
  memset(base_, 0, seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT);
  if (base_ == nullptr) {
    LOG(ERROR) << "Cannot init table data, not enough memory!";
    return -1;
  }
  uint64_t str_capacity = DOCNUM_PER_SEGMENT * 32 * string_field_num + 1;
  base_str_ = new (std::nothrow) char[str_capacity];
  if (base_str_ == nullptr) return -1;

  id_ = id;
  file_name_ = path + "/" + std::to_string(id_) + ".profile";
  int fd = open(file_name_.c_str(), O_RDWR | O_CREAT, 00666);
  if (-1 == fd) {
    LOG(ERROR) << "open vector file error, path=" << file_name_;
    return -1;
  }

  mmap_buf_ = (char *)mmap(NULL, MAX_MMAP_SIZE, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
  if ((void *)mmap_buf_ == MAP_FAILED) {
    LOG(ERROR) << "Mmap error";
    close(fd);
    return -1;
  }
  close(fd);
  int ret = Truncate(file_name_,
                     seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT);
  if (ret != 0) {
    return -1;
  }
  SetStrCapacity(str_capacity);

  str_file_name_ = path + "/" + std::to_string(id_) + ".str.profile";
  fd = open(str_file_name_.c_str(), O_RDWR | O_CREAT, 00666);
  if (-1 == fd) {
    LOG(ERROR) << "open vector file error, path=" << str_file_name_;
    return -1;
  }

  mmap_str_buf_ = (char *)mmap(NULL, MAX_MMAP_SIZE, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0);
  if ((void *)mmap_str_buf_ == MAP_FAILED) {
    LOG(ERROR) << "Mmap error";
    close(fd);
    return -1;
  }
  close(fd);
  ret = Truncate(str_file_name_, StrCapacity());
  if (ret != 0) {
    return -1;
  }

  ret = pthread_rwlock_init(&shared_mutex_, NULL);
  if (ret != 0) {
    LOG(ERROR) << "Mutex init failed";
  }

  SetVersion(version_);
  b_running_ = true;
  return ret;
}

int TableData::Truncate(std::string &path, off_t length) {
  if (truncate(path.c_str(), length)) {
    LOG(ERROR) << "truncate feature file=" << path << " to " << length
               << ", error:" << strerror(errno);
    return -1;
  }
  return 0;
}

char *TableData::Base() {
  char *base = nullptr;
  if (mode_ == TABLE_LOAD_MODE::MODE_DISK) {
    base = mmap_buf_ + seg_header_size_;
  } else {
    base = base_ + seg_header_size_;
  }
  return base;
}

int TableData::Write(const char *value, uint64_t offset, int len) {
  WriteThreadLock write_lock(shared_mutex_);
  memcpy(base_ + seg_header_size_ + offset, value, len);
  memcpy(mmap_buf_ + seg_header_size_ + offset, value, len);
  return 0;
}

int TableData::GetStr(IN str_offset_t offset, IN str_len_t len,
                      OUT std::string &str, DecompressStr &decompress_str) {
  if (offset > StrSize()) {
    LOG(ERROR) << "offset [" << offset << "] out of range [" << StrSize()
               << "]";
    return -1;
  }
  ReadThreadLock read_lock(shared_mutex_);
  char *base_str = base_str_;
  str_offset_t compressed_size = StrCompressedSize();
  if (BCompressed() == 0) {
    str = std::string(base_str + offset, len);
  } else {
    if (decompress_str.Hit()) {
      str = std::string(decompress_str.Str().c_str() + offset, len);
    } else {
      auto de_size = ZSTD_getDecompressedSize(base_str, compressed_size);
      char *de_char = new char[de_size];
      size_t size =
          ZSTD_decompress(de_char, de_size, base_str, compressed_size);
      size_t ret = ZSTD_isError(size);
      if (ret != 0) {
        LOG(ERROR) << "ZSTD_decompress error";
        delete[] de_char;
        return -1;
      }
      str = std::string(de_char + offset, len);
      decompress_str.SetStr(std::string(de_char, size));
      delete[] de_char;
    }
  }
  return 0;
}

int TableData::WriteStr(IN const char *str, IN str_len_t len) {
  WriteThreadLock write_lock(shared_mutex_);
  auto str_size = StrSize();
  if (BCompressed() == 0) {
    uint64_t str_capacity = StrCapacity();
    if (str_size + len >= str_capacity) {
      char *new_base_str = new (std::nothrow) char[str_capacity << 1];
      memcpy(new_base_str, base_str_, str_capacity);
      char *old = base_str_;
      base_str_ = new_base_str;
      delete[] old;

      int ret = Truncate(str_file_name_, str_capacity << 1);
      if (ret != 0) {
        return -1;
      }
      SetStrCapacity(str_capacity << 1);
    }

    memcpy(base_str_ + str_size, str, len);
    memcpy(mmap_str_buf_ + str_size, str, len);

    SetStrSize(str_size + len);
  } else {
    str_offset_t compressed_size = StrCompressedSize();
    auto de_size = ZSTD_getDecompressedSize(base_str_, compressed_size);
    char *de_char = new char[de_size + len];
    size_t size = ZSTD_decompress(de_char, de_size, base_str_, compressed_size);
    size_t ret = ZSTD_isError(size);
    if (ret != 0) {
      LOG(ERROR) << "ZSTD_decompress error";
      delete[] de_char;
      return -1;
    }
    memcpy(de_char + de_size, str, len);
    SetStrSize(str_size + len);
    SetStrCapacity(str_size + len);
    Compress(de_char);
    delete[] de_char;
  }
  return 0;
}

int TableData::WriteStr(IN const char *str, IN str_offset_t offset,
                        IN str_len_t len) {
  WriteThreadLock write_lock(shared_mutex_);
  if (BCompressed() == 0) {
    memcpy(base_str_ + offset, str, len);
    memcpy(mmap_str_buf_ + offset, str, len);
  } else {
    str_offset_t compressed_size = StrCompressedSize();
    auto de_size = ZSTD_getDecompressedSize(base_str_, compressed_size);
    char *de_char = new char[de_size];
    size_t size = ZSTD_decompress(de_char, de_size, base_str_, compressed_size);
    size_t ret = ZSTD_isError(size);
    if (ret != 0) {
      LOG(ERROR) << "ZSTD_decompress error";
      delete[] de_char;
      return -1;
    }
    memcpy(de_char + offset, str, len);
    Compress(de_char);
    delete[] de_char;
  }
  return 0;
}

void TableData::Compress(IN char *str) {
  auto str_size = StrSize();
  size_t dstCapacity = ZSTD_compressBound(str_size);
  char *compress_str = new char[dstCapacity];
  size_t size = ZSTD_compress(compress_str, dstCapacity, str, str_size, 1);
  size_t ret = ZSTD_isError(size);
  if (ret != 0) {
    LOG(ERROR) << "ZSTD_compress error";
    delete[] compress_str;
    return;
  }
  str_offset_t str_compressed_size = size;
  char *new_str = new char[str_compressed_size];
  memcpy(new_str, compress_str, str_compressed_size);
  char *old = base_str_;
  base_str_ = new_str;
  delete[] old;

  memcpy(mmap_str_buf_, compress_str, str_compressed_size);
  SetStrCompressedSize(str_compressed_size);
  delete[] compress_str;
}

int TableData::Compress() {
  WriteThreadLock write_lock(shared_mutex_);
  Compress(base_str_);
  SetStrCapacity(StrSize());
  SetCompressed(1);
  return 0;
}

int TableData::Load(int id, const std::string &path) {
  id_ = id;
  file_name_ = path + "/" + std::to_string(id_) + ".profile";
  int fd = open(file_name_.c_str(), O_RDWR, 00666);
  if (-1 == fd) {
    LOG(ERROR) << "open vector file error, path=" << file_name_;
    return -1;
  }

  mmap_buf_ = (char *)mmap(NULL, MAX_MMAP_SIZE, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
  if ((void *)mmap_buf_ == MAP_FAILED) {
    LOG(ERROR) << "Mmap error";
    close(fd);
    return -1;
  }
  close(fd);

  str_file_name_ = path + "/" + std::to_string(id_) + ".str.profile";
  fd = open(str_file_name_.c_str(), O_RDWR, 00666);
  if (-1 == fd) {
    LOG(WARNING) << "No string file" << str_file_name_;
  } else {
    mmap_str_buf_ = (char *)mmap(NULL, MAX_MMAP_SIZE, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
    if ((void *)mmap_str_buf_ == MAP_FAILED) {
      LOG(ERROR) << "Mmap error";
      close(fd);
      return -1;
    }
  }

  close(fd);
  int ret = pthread_rwlock_init(&shared_mutex_, NULL);
  if (ret != 0) {
    LOG(ERROR) << "Mutex init failed";
  }
  base_ = new (
      std::nothrow) char[seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT];
  memset(base_, 0, seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT);
  if (base_ == nullptr) {
    LOG(ERROR) << "Cannot init table data, not enough memory!";
    return -1;
  }

  memcpy(base_, mmap_buf_,
         seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT);

  uint64_t str_capacity = StrCapacity();
  base_str_ = new (std::nothrow) char[str_capacity];
  if (base_str_ == nullptr) return -1;

  memcpy(base_str_, mmap_str_buf_, str_capacity);

  return ret;
}

}  // namespace table
}  // namespace tig_gamma
