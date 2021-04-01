/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <unistd.h>
#include <vector>

#include "block.h"
#include "lru_cache.h"



typedef uint32_t str_offset_t;
typedef uint16_t str_len_t;

namespace tig_gamma {


class VectorBlock : public Block {
 public:
  VectorBlock(int fd, int per_block_size, int length, uint32_t header_size);

  void InitSubclass() override;

  int GetReadFunParameter(ReadFunParameter &parameter) override;

  static bool ReadBlock(uint64_t key, std::shared_ptr<std::vector<uint8_t>> &block,
                        ReadFunParameter *param);

  int WriteContent(const uint8_t *data, int len, uint32_t offset,
                   disk_io::AsyncWriter *disk_io) override;

  int ReadContent(uint8_t *value, uint32_t len, uint32_t offset) override;

  int SubclassUpdate(const uint8_t *data, int len, uint32_t offset) override;
 private:
  int vec_item_len_;
};

}  // namespace tig_gamma
