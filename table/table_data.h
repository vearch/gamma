/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits.h>
#include <stdio.h>

#include <string>

#include "table_define.h"

namespace table {

enum class TABLE_LOAD_MODE : std::uint16_t {
  MODE_FULL_MEMORY = 1,
  MODE_DISK = 2
};

class TableData {
 public:
  TableData(int item_length);
  ~TableData();

  int Init();

  char *Base();

  int GetStr(IN str_offset_t offset, IN str_len_t len, OUT std::string &str,
             DecompressStr &decompress_str);

  int WriteStr(IN const char *str, IN str_len_t len);

  int WriteStr(IN const char *str, IN str_offset_t offset, IN str_len_t len);

  str_offset_t StrOffset() { return str_size_; }

  int Compress();

 private:
  char *Compress(IN char *str);
  int close();

 protected:
  char *base_;
  char *base_str_;
  str_offset_t str_size_;
  str_offset_t str_compressed_size_;
  uint64_t str_capacity_;

  uint32_t capacity_;
  uint32_t size_;

  uint64_t seg_header_size_;

  uint32_t item_length_;
  bool b_compressed_;
};

}  // namespace table