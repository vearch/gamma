/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "block.h"

namespace tig_gamma {

class TableBlock : public Block {
 public:
  TableBlock(int fd, int max_size, int length, uint32_t header_size);

  void InitSubclass() {};

  int GetReadFunParameter(ReadFunParameter &parameter);

  static bool ReadBlock(uint64_t key, std::shared_ptr<std::vector<uint8_t>> &block,
                        ReadFunParameter *param);

  int WriteContent(const uint8_t *data, int len, uint32_t offset,
                   disk_io::AsyncWriter *disk_io) override;

  int Add(const uint8_t *data, int len);

  int ReadContent(uint8_t *value, uint32_t len, uint32_t offset) override;

  int SubclassUpdate(const uint8_t *data, int len, uint32_t offset) override;
};

}  // namespace tig_gamma