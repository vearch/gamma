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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <zstd.h>

#include <new>

#include "log.h"
#include "utils.h"

namespace table {

TableData::TableData(int item_length) {
  base_ = nullptr;
  base_str_ = nullptr;
  item_length_ = item_length;
  str_size_ = 0;
  str_capacity_ = 0;
  capacity_ = 0;
  size_ = 0;
  str_compressed_size_ = 0;
  b_compressed_ = false;
  seg_header_size_ = sizeof(capacity_) + sizeof(size_);
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
}

int TableData::Init() {
  base_ = new (
      std::nothrow) char[seg_header_size_ + item_length_ * DOCNUM_PER_SEGMENT];
  if (base_ == nullptr) {
    LOG(ERROR) << "Cannot init table data, not enough memory!";
    return -1;
  }
  str_capacity_ = DOCNUM_PER_SEGMENT * 32;
  base_str_ = new (std::nothrow) char[str_capacity_];
  if (base_ == nullptr) return -1;
  return 0;
}

static void FreeOldData(char *data) { delete data; }

char *TableData::Base() { return base_ + seg_header_size_; }

int TableData::GetStr(IN str_offset_t offset, IN str_len_t len,
                      OUT std::string &str, DecompressStr &decompress_str) {
  if (offset > str_size_) {
    LOG(ERROR) << "offset [" << offset << "] out of range [" << str_size_
               << "]";
    return -1;
  }
  if (not b_compressed_) {
    str = std::string(base_str_ + offset, len);
  } else {
    // auto de_size = ZSTD_getDecompressedSize(base_str_, str_compressed_size_);
    if (decompress_str.Hit()) {
      str = std::string(decompress_str.Str().c_str() + offset, len);
    } else {
      auto de_size = str_capacity_;
      char *de_char = new char[de_size];
      size_t size =
          ZSTD_decompress(de_char, de_size, base_str_, str_compressed_size_);
      str = std::string(de_char + offset, len);
      decompress_str.SetStr(std::string(de_char, size));
      delete[] de_char;
    }
  }
  return 0;
}

int TableData::WriteStr(IN const char *str, IN str_len_t len) {
  if (not b_compressed_) {
    if (str_size_ + len >= str_capacity_) {
      char *new_base_str = new (std::nothrow) char[str_capacity_ << 1];
      memcpy(new_base_str, base_str_, str_capacity_);
      str_capacity_ = str_capacity_ << 1;
      char *old = base_str_;
      base_str_ = new_base_str;
      auto func_free = std::bind(FreeOldData, std::placeholders::_1);

      utils::AsyncWait(1000, func_free, old);
    }

    memcpy(base_str_ + str_size_, str, len);

    str_size_ += len;
  } else {
    auto de_size = str_capacity_;
    char *de_char = new char[de_size + len];
    ZSTD_decompress(de_char, de_size, base_str_, str_compressed_size_);
    memcpy(de_char + de_size, str, len);
    str_size_ += len;
    str_capacity_ = str_size_;
    char *old = base_str_;
    base_str_ = Compress(de_char);
    auto func_free = std::bind(FreeOldData, std::placeholders::_1);

    utils::AsyncWait(1000, func_free, old);
  }
  return 0;
}

int TableData::WriteStr(IN const char *str, IN str_offset_t offset,
                        IN str_len_t len) {
  if (not b_compressed_) {
    memcpy(base_str_ + offset, str, len);
  } else {
    auto de_size = str_capacity_;
    char *de_char = new char[de_size];
    ZSTD_decompress(de_char, de_size, base_str_, str_compressed_size_);
    memcpy(de_char + offset, str, len);
    char *old = base_str_;
    base_str_ = Compress(de_char);
    delete[] de_char;
    auto func_free = std::bind(FreeOldData, std::placeholders::_1);

    utils::AsyncWait(1000, func_free, old);
  }
  return 0;
}

char *TableData::Compress(IN char *str) {
  size_t dstCapacity = ZSTD_compressBound(str_size_);
  char *compress_str = new char[dstCapacity];
  str_compressed_size_ =
      ZSTD_compress(compress_str, dstCapacity, str, str_size_, 1);
  char *new_str = new char[str_compressed_size_];
  memcpy(new_str, compress_str, str_compressed_size_);
  delete[] compress_str;
  return new_str;
}

int TableData::Compress() {
  char *old = base_str_;
  base_str_ = Compress(base_str_);
  str_capacity_ = str_size_;
  b_compressed_ = true;
  auto func_free = std::bind(FreeOldData, std::placeholders::_1);

  utils::AsyncWait(1000, func_free, old);
  return 0;
}
}  // namespace table
