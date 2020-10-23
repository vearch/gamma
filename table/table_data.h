/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits.h>
#include <stdio.h>

#include <pthread.h>
#include <string>

#include "table_define.h"

namespace tig_gamma {
namespace table {

enum class TABLE_LOAD_MODE : std::uint16_t {
  MODE_FULL_MEMORY = 1,
  MODE_DISK = 2,
  MODE_MEMORY_DISK = 3
};

class TableData {
 public:
  TableData(int item_length);
  ~TableData();

  int Init(int id, const std::string &path, uint8_t string_field_num);

  char *Base();

  int Write(const char *value, uint64_t offset, int len);

  int GetStr(IN str_offset_t offset, IN str_len_t len, OUT std::string &str,
             DecompressStr &decompress_str);

  int WriteStr(IN const char *str, IN str_len_t len);

  int WriteStr(IN const char *str, IN str_offset_t offset, IN str_len_t len);

  str_offset_t StrOffset() { return StrSize(); }

  int Compress();

  int Load(int id, const std::string &path);

  uint8_t Version(); 

  void SetVersion(uint8_t version);

  uint32_t Size();

  void SetSize(uint32_t size);

  uint64_t StrCapacity();

  void SetStrCapacity(uint64_t str_capacity);

  str_offset_t StrSize();

  void SetStrSize(str_offset_t str_size);

  uint8_t BCompressed();

  void SetCompressed(uint8_t compressed);

  str_offset_t StrCompressedSize();

  void SetStrCompressedSize(str_offset_t str_compressed_size);

  int Sync();

 private:

  int Truncate(std::string &path, off_t length);

  void Compress(IN char *str);

  int Close();

 protected:

  uint32_t capacity_;

  uint64_t seg_header_size_;
  uint32_t seg_header_backup_;
  uint8_t version_;

  uint32_t item_length_;

  char *base_;
  char *base_str_;

  char *mmap_buf_;
  char *mmap_str_buf_;
  std::string file_name_;
  std::string str_file_name_;
  int id_;

  bool b_running_;

  TABLE_LOAD_MODE mode_;
  pthread_rwlock_t shared_mutex_;
};

}  // namespace table
}