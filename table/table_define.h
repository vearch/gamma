/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace table {

#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#define TABLE_MAIN "table.main"
#define TABLE_EXT "table.ext"

const static int DOCNUM_PER_SEGMENT = 1 << 10;  // 1024
const static int MAX_SEGMENT_NUM = 102400;      // max segment num

typedef uint32_t str_offset_t;
typedef uint8_t str_len_t;

class DecompressStr {
 public:
  DecompressStr() {
    segid_ = -1;
    hit_ = false;
  }

  std::string &Str() { return str_; }

  void SetStr(const std::string &str) { str_ = str; }

  void SetStr(std::string &&str) { str_ = std::move(str); }

  bool Hit() { return hit_; }

  void SetHit(bool hit) { hit_ = hit; }

  int SegID() { return segid_; }

  void SetSegID(int segid) { segid_ = segid; }

 private:
  std::string str_;
  int segid_;
  bool hit_;
};

}  // namespace table