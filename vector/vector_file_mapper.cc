/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "vector_file_mapper.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error_code.h"
#include "log.h"
#include "utils.h"
#include "error_code.h"

namespace tig_gamma {

VectorFileMapper::VectorFileMapper(const std::string &file_path,
                                   int max_vec_size, int vec_byte_size)
    : file_path_(file_path),
      max_vec_size_(max_vec_size),
      vec_byte_size_(vec_byte_size) {
  mapped_byte_size_ = (size_t)max_vec_size * vec_byte_size_;
  vectors_ = nullptr;
  curr_idx_ = 0;
}

VectorFileMapper::~VectorFileMapper() {
  if (vectors_ != nullptr) {
    int ret = munmap(vectors_, mapped_byte_size_);
    if (ret != 0) {
      LOG(ERROR) << "munmap error: " << strerror(errno) << ", ret=" << ret;
    }
    vectors_ = nullptr;
  }
}

int VectorFileMapper::Init() {
  int fd = open(file_path_.c_str(), O_RDWR | O_CREAT, 0666);
  if (-1 == fd) {
    LOG(ERROR) << "open vector file error, path=" << file_path_;
    return IO_ERR;
  }

  if (ftruncate(fd, mapped_byte_size_)) {
    close(fd);
    LOG(ERROR) << "truncate file error:" << strerror(errno);
    return IO_ERR;
  }

  vectors_ = (uint8_t *)mmap(NULL, mapped_byte_size_, PROT_READ | PROT_WRITE,
                             MAP_SHARED, fd, 0);
  close(fd);
  if (vectors_ == MAP_FAILED) {
    LOG(ERROR) << "mmap error:" << strerror(errno);
    return INTERNAL_ERR;
  }

  int ret = madvise(vectors_, mapped_byte_size_, MADV_RANDOM);
  if (ret != 0) {
    LOG(ERROR) << "madvise error: " << strerror(errno) << ", ret=" << ret;
    return INTERNAL_ERR;
  }
  LOG(INFO) << "map success! max byte size=" << mapped_byte_size_
            << ", file path=" << file_path_;
  return 0;
}

int VectorFileMapper::Add(uint8_t *vec, int len) {
  memcpy(vectors_ + (size_t)curr_idx_ * vec_byte_size_, vec, len);
  ++curr_idx_;
  return 0;
}

const uint8_t *VectorFileMapper::GetVector(int id) {
  return vectors_ + ((long)id) * vec_byte_size_;
}

const uint8_t *VectorFileMapper::GetVectors() { return vectors_; }

int VectorFileMapper::Update(int vid, uint8_t *vec, int len) {
  assert(vec_byte_size_ == len);
  memcpy(vectors_ + (size_t)vid * vec_byte_size_, vec, vec_byte_size_);
  return 0;
}
int VectorFileMapper::Sync() {
  if (msync(vectors_, mapped_byte_size_, MS_SYNC)) {
    LOG(ERROR) << "msync error: " << strerror(errno);
    return IO_ERR;
  }
  return 0;
}

}  // namespace tig_gamma
