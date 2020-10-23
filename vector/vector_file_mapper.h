/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/mman.h>

#include <string>

namespace tig_gamma {

class VectorFileMapper {
 public:
  VectorFileMapper(const std::string &file_path, int max_vec_size,
                   int vec_byte_size);

  ~VectorFileMapper();

  int Init();

  int Add(uint8_t *vec, int len);

  const uint8_t *GetVector(int id);

  const uint8_t *GetVectors();

  bool IsFull() { return curr_idx_ == max_vec_size_; }

  int Sync();

  void SetCurrIdx(int curr_idx) { curr_idx_ = curr_idx; }

  int Update(int vid, uint8_t *vec, int len);

 private:
  uint8_t *vectors_;
  std::string file_path_;
  size_t mapped_byte_size_;

  int max_vec_size_;
  int vec_byte_size_;
  int curr_idx_;
};

}  // namespace tig_gamma
