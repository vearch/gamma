/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include "block.h"

namespace tig_gamma {

class StringBlock : public Block {
 public:
  StringBlock(int fd, int max_size, int length, uint32_t header_size,
              uint32_t seg_id, uint32_t seg_block_capacity_);

  ~StringBlock();

  int GetReadFunParameter(ReadFunParameter &parameter, uint32_t len, 
                          uint32_t off) { return 0; };
  
  int LoadIndex(const std::string &file_path);

  int CloseBlockPosFile();

  int WriteContent(const uint8_t *data, int len, uint32_t offset,
                   disk_io::AsyncWriter *disk_io);

  void InitStrBlock(void *lru);
  
  int Add(const uint8_t *data, int len);

  int ReadContent(uint8_t *value, uint32_t len, uint32_t offset);

  int WriteString(const char *data, str_len_t len, str_offset_t offset,
                  uint32_t &block_id, uint32_t &in_block_pos);

  int Read(uint32_t block_id, uint32_t in_block_pos, str_len_t len,
           std::string &str_out);

  static bool ReadString(uint32_t key,
                         std::shared_ptr<std::vector<uint8_t>> &block,
                         ReadStrFunParameter *param);

 private:
  void InitSubclass() {};

  int SubclassUpdate(const uint8_t *data, int len, uint32_t offset) {
    return 0;
  };

  int AddBlockPos(uint32_t block_pos);

  LRUCache<uint32_t, std::vector<uint8_t>, ReadStrFunParameter *> *lru_cache_;

  std::string block_pos_file_path_;

  FILE *block_pos_fp_;

  tbb::concurrent_vector<uint32_t> block_pos_;  // <block id, offset>

};

}  // namespace tig_gamma