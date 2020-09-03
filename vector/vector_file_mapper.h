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
  VectorFileMapper(std::string &file_path, int max_vector_size, int dimension,
                   uint8_t data_size);

  ~VectorFileMapper();

  int Init();

  const uint8_t *GetVector(int id);

  const uint8_t *GetVectors();

  int GetMappedNum() const { return mapped_num_; };

 private:
  void *buf_;
  uint8_t *vectors_;
  std::string file_path_;
  int offset_;
  int dimension_;
  size_t mapped_byte_size_;
  int mapped_num_;

  uint8_t data_size_;
};

}  // namespace tig_gamma
