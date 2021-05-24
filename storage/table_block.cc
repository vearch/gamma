/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "table_block.h"

#include <string.h>
#include <unistd.h>

namespace tig_gamma {

TableBlock::TableBlock(int fd, int per_block_size, int length,
                       uint32_t header_size, uint32_t seg_id,
                       uint32_t seg_block_capacity,
                       const std::atomic<uint32_t> *cur_size, int max_size)
    : Block(fd, per_block_size, length, header_size, seg_id,
            seg_block_capacity, cur_size, max_size) {}

int TableBlock::GetReadFunParameter(ReadFunParameter &parameter, uint32_t len,
                                     uint32_t off) {
  parameter.fd = fd_;
  parameter.len = len;
  parameter.offset = off + header_size_;
  return 0;
}

int TableBlock::WriteContent(const uint8_t *data, int len, uint32_t offset,
                             disk_io::AsyncWriter *disk_io,
                             std::atomic<uint32_t> *cur_size) {
  disk_io->Set(header_size_, item_length_);
  struct disk_io::WriterStruct *write_struct = new struct disk_io::WriterStruct;
  write_struct->fd = fd_;
  write_struct->data = new uint8_t[len];
  memcpy(write_struct->data, data, len);
  write_struct->start = header_size_ + offset;
  write_struct->len = len;
  write_struct->cur_size = cur_size;
  disk_io->AsyncWrite(write_struct);
  // disk_io->SyncWrite(write_struct);
  return 0;
}

bool TableBlock::ReadBlock(uint32_t key, char *block,
                           ReadFunParameter *param) {
  if (param->len > MAX_BLOCK_SIZE) {
    LOG(ERROR) << "Tableblock read len:" << param->len
               << " key:" << key;
    return false;
  }
  if (block == nullptr) {
    LOG(ERROR) << "ReadString block is nullptr.";
    return false;
  }
  pread(param->fd, block, param->len, param->offset);
  return true;
}

int TableBlock::ReadContent(uint8_t *value, uint32_t len, uint32_t offset) {
  pread(fd_, value, len, header_size_ + offset);
  return 0;
}

}  // namespace tig_gamma
