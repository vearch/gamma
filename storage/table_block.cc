/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "table_block.h"

#include <string.h>
#include <unistd.h>
// #include <concurrent_vector.h>

namespace tig_gamma {

TableBlock::TableBlock(int fd, int per_block_size, int length,
                       uint32_t header_size)
    : Block(fd, per_block_size, length, header_size) {}

int TableBlock::GetReadFunParameter(ReadFunParameter &parameter) {
  parameter.offset += header_size_;
  return 0;
}


int TableBlock::WriteContent(const uint8_t *data, int len, uint32_t offset,
                             disk_io::AsyncWriter *disk_io) {
  disk_io->Set(header_size_, item_length_);
  struct disk_io::WriterStruct *write_struct = new struct disk_io::WriterStruct;
  write_struct->fd = fd_;
  write_struct->data = new uint8_t[len];
  memcpy(write_struct->data, data, len);
  write_struct->start = header_size_ + offset;
  write_struct->len = len;
  disk_io->AsyncWrite(write_struct);
  // disk_io->SyncWrite(write_struct);
  return 0;
}

bool TableBlock::ReadBlock(uint64_t key,
                           std::shared_ptr<std::vector<uint8_t>> &block,
                           ReadFunParameter *param) {
  block = std::make_shared<std::vector<uint8_t>>(param->len);
  pread(param->fd, block->data(), param->len, param->offset);
  return true;
}

int TableBlock::ReadContent(uint8_t *value, uint32_t len, uint32_t offset) {
  pread(fd_, value, len, header_size_ + offset);
  return 0;
}

int TableBlock::SubclassUpdate(const uint8_t *data, int len, uint32_t offset) {
  pwrite(fd_, data, len, header_size_ + offset);
  return 0;
}

}  // namespace tig_gamma
