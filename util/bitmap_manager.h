/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

#include "util/log.h"

namespace bitmap {

template <typename T>
class BitmapManager {
 public:
  BitmapManager();
  ~BitmapManager();

  int Init(T bit_size, std::string fpath = "", char *bitmap = nullptr);

  int SetDumpFilePath(std::string fpath);

  int DumpBitmap(T begin_bit_id = 0, T bit_len = 0);

  int LoadBitmap(T begin_bit_id = 0, T bit_len = 0);

  T GetBitmapFileBytesSize();

  int SetN(T bit_id);

  int UnsetN(T bit_id);

  bool GetN(T bit_id);

  T BitSize() { return bit_size_; }

  char *Bitmap() { return bitmap_; }

  T BytesSize() { return (bit_size_ >> 3) + 1; }

 private:
  char *bitmap_;
  T bit_size_;
  int fd_;
  std::string fpath_;
};

template <typename T>
BitmapManager<T>::BitmapManager() {
  bitmap_ = nullptr;
  bit_size_ = 0;
  fd_ = -1;
  fpath_ = "";
}

template <typename T>
BitmapManager<T>::~BitmapManager() {
  if (bitmap_) {
    delete[] bitmap_;
    bitmap_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

template <typename T>
int BitmapManager<T>::Init(T bit_size, std::string fpath, char *bitmap) {
  if (bit_size <= 0) {
    LOG(INFO) << "bit_size <= 0";
    return -1;
  }
  this->bit_size_ = bit_size;
  T bytes_count = (bit_size >> 3) + 1;

  if (bitmap) {
    bitmap_ = bitmap;
  } else {
    bitmap_ = new char[bytes_count];
    if (bitmap_ == nullptr) {
      LOG(INFO) << "new char[" << bytes_count << "] error.";
      return -1;
    }
  }
  memset(bitmap_, 0, bytes_count);

  // open dump file
  int ret = 0;
  if (not fpath.empty() && fd_ == -1) {
    fpath_ = fpath;
    fd_ = open(fpath_.c_str(), O_RDWR | O_CREAT, 0666);
    if (-1 == fd_) {
      LOG(ERROR) << "open file error, path=" << fpath_;
      ret = -1;
    }
  }
  LOG(INFO) << "BitmapManager init successed. bytes_count=" << bytes_count
            << " bit_size=" << bit_size;
  return ret;
}

template <typename T>
int BitmapManager<T>::SetDumpFilePath(std::string fpath) {
  if (not fpath.empty()) {
    if (fd_ != -1) {
      LOG(ERROR) << "The file[" << fpath_ << "] is already open. close it.";
      close(fd_);
    }
    fpath_ = fpath;
    fd_ = open(fpath_.c_str(), O_RDWR | O_CREAT, 0666);
    if (-1 == fd_) {
      LOG(ERROR) << "open file error, path=" << fpath_;
      return -1;
    }
    LOG(INFO) << "open bitmap file[" << fpath << "] success.";
    return 0;
  }
  return -1;
}

template <typename T>
int BitmapManager<T>::DumpBitmap(T begin_bit_id, T bit_len) {
  if (bit_len == 0) bit_len = bit_size_;
  if (begin_bit_id < 0 || bit_len < 0 || begin_bit_id + bit_len > bit_size_) {
    LOG(ERROR) << "parameters error, begin_bit_id=" << begin_bit_id
               << " dump_bit_len=" << bit_len << " bit_size=" << bit_size_;
    return -1;
  }

  T begin_bytes = begin_bit_id >> 3;
  T end_bytes = (begin_bit_id + bit_len - 1) >> 3;
  T dump_bytes = end_bytes - begin_bytes + 1;
  int ret = 0;
  if (fd_ != -1) {
    T written_bytes = 0;
    int i = 0;
    while (written_bytes < dump_bytes) {
      T bytes = pwrite(fd_, bitmap_ + begin_bytes + written_bytes,
                       dump_bytes - written_bytes, begin_bytes + written_bytes);
      written_bytes += bytes;
      if (++i >= 1000) {
        LOG(ERROR) << "dumped bitmap is not complate, written_bytes="
                   << written_bytes;
        ret = -1;
        break;
      }
    }
  } else {
    ret = -1;
  }
  return ret;
}

template <typename T>
int BitmapManager<T>::LoadBitmap(T begin_bit_id, T bit_len) {
  if (bit_len == 0) bit_len = bit_size_;
  if (begin_bit_id < 0 || bit_len < 0 || begin_bit_id + bit_len > bit_size_) {
    LOG(ERROR) << "parameters error, begin_bit_id=" << begin_bit_id
               << " load_bit_len=" << bit_len << " bit_size=" << bit_size_;
    return -1;
  }

  T begin_bytes = begin_bit_id >> 3;
  T end_bytes = (begin_bit_id + bit_len - 1) >> 3;
  T load_bytes = end_bytes - begin_bytes + 1;
  int ret = 0;
  if (fd_ != -1) {
    T read_bytes = 0;
    int i = 0;
    while (read_bytes < load_bytes) {
      T bytes = pread(fd_, bitmap_ + begin_bytes + read_bytes,
                      load_bytes - read_bytes, begin_bytes + read_bytes);
      read_bytes += bytes;
      if (++i >= 1000) {
        LOG(ERROR) << "load bitmap is not complate, load_bytes=" << read_bytes;
        ret = -1;
        break;
      }
    }
  } else {
    ret = -1;
  }
  return ret;
}

template <typename T>
T BitmapManager<T>::GetBitmapFileBytesSize() {
  if (fd_ != -1) {
    T len = lseek(fd_, 0, SEEK_END);
    return len;
  }
  return -1;
}

template <typename T>
int BitmapManager<T>::SetN(T bit_id) {
  if (bit_id >= 0 && bit_id < bit_size_ && bitmap_ != nullptr) {
    bitmap_[bit_id >> 3] |= (0x1 << (bit_id & 0x7));
    return 0;
  }
  return -1;
}

template <typename T>
int BitmapManager<T>::UnsetN(T bit_id) {
  if (bit_id >= 0 && bit_id < bit_size_ && bitmap_ != nullptr) {
    bitmap_[bit_id >> 3] &= ~(0x1 << (bit_id & 0x7));
    return 0;
  }
  return -1;
}

template <typename T>
bool BitmapManager<T>::GetN(T bit_id) {
  if (bit_id >= 0 && bit_id < bit_size_ && bitmap_ != nullptr) {
    return (bitmap_[bit_id >> 3] & (0x1 << (bit_id & 0x7)));
  }
  return false;
}

}  // namespace bitmap
