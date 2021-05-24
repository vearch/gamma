/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "async_writer.h"

#include <unistd.h>

#include "log.h"

namespace tig_gamma {
namespace disk_io {

AsyncWriter::AsyncWriter() {
  running_ = true;
  writer_q_ = new WriterQueue;
  auto func_operate = std::bind(&AsyncWriter::WriterHandler, this);
  handler_thread_ = std::thread(func_operate);
}

AsyncWriter::~AsyncWriter() {
  Sync();
  running_ = false;
  handler_thread_.join();
  delete writer_q_;
  writer_q_ = nullptr;
}

static uint32_t WritenSize(int fd) {
  uint32_t size;
  pread(fd, &size, sizeof(size), sizeof(uint8_t) + sizeof(uint32_t));
  return size;
}

static void UpdateSize(int fd, int num) {
  uint32_t cur_size = WritenSize(fd) + num;
  pwrite(fd, &cur_size, sizeof(cur_size), sizeof(uint8_t) + sizeof(uint32_t));
}

int AsyncWriter::WriterHandler() {
  int bulk_size = 1000;
  int bulk_bytes = 64 * 1024 * 2048;  // TODO check overflow
  uint8_t *buffer = new uint8_t[bulk_bytes];

  while (running_) {
    struct WriterStruct *writer_structs[bulk_size];

    int size = 0;
    while(not writer_q_->empty() && size < bulk_size) {
      struct WriterStruct *pop_val = nullptr;
      bool ret = writer_q_->try_pop(pop_val);
      if (ret) writer_structs[size++] = pop_val;
    }

    if (size > 1) {
      int fd = -1;
      int prev_fd = writer_structs[0]->fd;

      uint32_t buffered_size = 0;
      uint32_t buffered_start = writer_structs[0]->start;

      for (size_t i = 0; i < size; ++i) {
        fd = writer_structs[i]->fd;
        uint8_t *data = writer_structs[i]->data;
        uint32_t len = writer_structs[i]->len;
        uint32_t start = writer_structs[i]->start;

        if (prev_fd != fd) {
          // flush prev data
          pwrite(prev_fd, buffer, buffered_size, buffered_start);
          UpdateSize(prev_fd, buffered_size / item_length_);
          prev_fd = fd;
          buffered_start = start;
          buffered_size = 0;
          // TODO check buffered_size + len < bulk_bytes
          memcpy(buffer + buffered_size, data, len);
          buffered_size += len;
        } else {
          if (buffered_size + len < bulk_bytes) {
            memcpy(buffer + buffered_size, data, len);
            buffered_size += len;
          } else {
            buffered_size += len;
            pwrite(fd, buffer, buffered_size, buffered_start);
            UpdateSize(fd, buffered_size / item_length_);
            buffered_size = 0;
            buffered_start = start;
          }
        }

        delete[] data;
        delete writer_structs[i];
      }
      pwrite(fd, buffer, buffered_size, buffered_start);
      UpdateSize(fd, buffered_size / item_length_);
      buffered_size = 0;
    } else if (size == 1) {
      int fd = writer_structs[0]->fd;
      uint8_t *data = writer_structs[0]->data;
      uint32_t start = writer_structs[0]->start;
      uint32_t len = writer_structs[0]->len;

      pwrite(fd, data, len, start);
      UpdateSize(fd, len / item_length_);

      delete[] data;
      delete writer_structs[0];
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // if (size < bulk_size) {
    //   std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // }
  }
  delete buffer;
  return 0;
}

int AsyncWriter::AsyncWrite(struct WriterStruct *writer_struct) {
  auto qu_size = writer_q_->size();
  while (qu_size > 10000) {
    LOG(INFO) << "AsyncWriter queue size[" << qu_size << "] > 10000, sleep 10ms";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    qu_size = writer_q_->size();
  }
  writer_q_->push(writer_struct);
  return 0;
}

int AsyncWriter::SyncWrite(struct WriterStruct *writer_struct) {
  int fd = writer_struct->fd;
  uint8_t *data = writer_struct->data;
  uint32_t start = writer_struct->start;
  uint32_t len = writer_struct->len;

  pwrite(fd, data, len, start);
  UpdateSize(fd, len / item_length_);

  delete data;
  delete writer_struct;
  return 0;
}

int AsyncWriter::Sync() {
  while (writer_q_->size()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
}

}  // namespace disk_io
}  // namespace tig_gamma