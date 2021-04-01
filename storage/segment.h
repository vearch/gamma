/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pthread.h>

#include <string>

#include "block.h"
#include "string_block.h"

namespace tig_gamma {

const static int MAX_SEGMENT_NUM = 102400;     // max segment num

class Segment {
 public:
  Segment(const std::string &file_path, int max_size, int vec_byte_size,
          disk_io::AsyncWriter *disk_io, void *table_cache, void *str_cache);

  ~Segment();

  int Init(BlockType block_type, Compressor *compressor = nullptr);

  int Load(BlockType block_type, Compressor *compressor = nullptr);

  int Add(const uint8_t *vec, int len);

  str_offset_t AddString(const char *vec, int len, uint32_t &block_id,
                         uint32_t &in_block_pos);

  int GetValue(uint8_t *value, int id);

  int GetValues(uint8_t *value, int id, int size);

  std::string GetString(uint32_t block_id, uint32_t in_block_pos,
                        str_len_t len);

  bool IsFull();

  void SetCurrIdx(int curr_idx) { SetSize(curr_idx); }

  int Update(int id, uint8_t *vec, int len);

 private:
  uint8_t Version();

  void SetVersion(uint8_t version);

  uint32_t BufferedSize();

  void PersistentedSize();

  void SetSize(uint32_t size);

  uint64_t StrCapacity();

  void SetStrCapacity(uint64_t str_capacity);

  uint32_t StrBlocksSize();

  void SetStrBlocksSize(uint32_t str_blocks_size);

  void SetBlocksStrSize(uint32_t str_blocks_size);

  str_offset_t StrSize();

  void SetStrSize(str_offset_t str_size);

  uint8_t BCompressed();

  void SetCompressed(uint8_t compressed);

  str_offset_t StrCompressedSize();

  void SetStrCompressedSize(str_offset_t str_compressed_size);

  int OpenFile();

  int InitBlock(BlockType block_type, Compressor *compressor);

 private:
  std::string file_path_;
  size_t mapped_byte_size_;

  int max_size_;
  uint32_t cur_size_;

  uint32_t buffered_size_;

  uint32_t capacity_;

  uint64_t seg_header_size_;
  uint8_t version_;

  uint32_t item_length_;

  int base_fd_;
  int str_fd_;

  Block *blocks_;

  StringBlock *str_blocks_;

  uint32_t per_block_size_;
  disk_io::AsyncWriter *disk_io_;

  void *cache_;
  void *str_cache_;
};

}  // namespace tig_gamma
