/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "mmap_raw_vector.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <exception>
#include "error_code.h"
#include "log.h"
#include "utils.h"

using namespace std;

namespace tig_gamma {

MmapRawVector::MmapRawVector(VectorMetaInfo *meta_info, const string &root_path,
                             const StoreParams &store_params,
                             const char *docids_bitmap)
    : RawVector(meta_info, root_path, docids_bitmap, store_params),
      AsyncFlusher(meta_info->Name()) {
  flush_batch_size_ = 1000;
  data_size_ = meta_info_->DataSize();
  vector_byte_size_ = data_size_ * meta_info->Dimension();
  flush_write_retry_ = 10;
  buffer_chunk_num_ = kDefaultBufferChunkNum;
  fet_fd_ = -1;
  max_size_ = store_params.segment_size_;
  max_buffer_size_ = 0;
}

MmapRawVector::~MmapRawVector() {
  CHECK_DELETE(vector_buffer_queue_);
  CHECK_DELETE(vector_file_mapper_);
  CHECK_DELETE_ARRAY(flush_batch_vectors_);
  if (fet_fd_ != -1) {
    fsync(fet_fd_);
    close(fet_fd_);
  }
}

int MmapRawVector::InitStore() {
  int dimension = meta_info_->Dimension();
  std::string &name = meta_info_->Name();

  std::string vec_dir = root_path_ + "/" + name;
  if (utils::make_dir(vec_dir.c_str())) {
    LOG(ERROR) << "mkdir error, path=" << vec_dir;
    return IO_ERR;
  }
  fet_file_path_ = vec_dir + "/vector.dat";
  fet_fd_ = open(fet_file_path_.c_str(), O_WRONLY | O_APPEND | O_CREAT, 00664);
  if (fet_fd_ == -1) {
    LOG(ERROR) << "open file error:" << strerror(errno);
    return IO_ERR;
  }

  max_buffer_size_ =
      (int)(this->store_params_->cache_size_ / this->vector_byte_size_);
  int remainder = max_buffer_size_ % buffer_chunk_num_;
  if (remainder > 0) {
    max_buffer_size_ += buffer_chunk_num_ - remainder;
  }

  vector_buffer_queue_ = new VectorBufferQueue(max_buffer_size_, dimension,
                                               buffer_chunk_num_, data_size_);
  vector_file_mapper_ = new VectorFileMapper(fet_file_path_, 1000 * 1000 * 10,
                                             dimension, data_size_);

  int ret = vector_buffer_queue_->Init();
  if (ret) {
    LOG(ERROR) << "init vector buffer queue error, ret=" << ret;
    return ret;
  }
  total_mem_bytes_ += vector_buffer_queue_->GetTotalMemBytes();

  flush_batch_vectors_ =
      new uint8_t[(uint64_t)flush_batch_size_ * vector_byte_size_];
  total_mem_bytes_ += (uint64_t)flush_batch_size_ * vector_byte_size_;

  ret = vector_file_mapper_->Init();
  if (ret) {
    LOG(ERROR) << "vector file mapper map error";
    return ret;
  }

  LOG(INFO) << "Init success! vector byte size=" << vector_byte_size_
            << ", flush batch size=" << flush_batch_size_
            << ", dimension=" << dimension << ", ntotal=" << meta_info_->Size()
            << ", init max_size=" << max_size_;
  return 0;
}

int MmapRawVector::FlushOnce() {
  int psize = vector_buffer_queue_->GetPopSize();
  int count = 0;

  while (count < psize) {
    int num =
        psize - count > flush_batch_size_ ? flush_batch_size_ : psize - count;
    vector_buffer_queue_->Pop(flush_batch_vectors_, vector_byte_size_, num, -1);
    ssize_t write_size = (ssize_t)num * vector_byte_size_;
    ssize_t ret = utils::write_n(fet_fd_, (char *)flush_batch_vectors_,
                                 write_size, flush_write_retry_);
    if (ret != write_size) {
      LOG(ERROR) << "write_n error:" << strerror(errno) << ", num=" << num;
      // TODO: truncate and seek file, or write the success number to file
      return -2;
    }
    count += num;
  }
  return psize;
}

int MmapRawVector::DumpVectors(int dump_vid, int n) {
  int dump_end = dump_vid + n;
  while (nflushed_ < dump_end) {
    LOG(INFO) << "raw vector=" << meta_info_->Name()
              << ", dump waiting! dump_end=" << dump_end
              << ", nflushed=" << nflushed_;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return SUCC;
}

int MmapRawVector::LoadVectors(int vec_num) {
  int dimension = meta_info_->Dimension();
  StopFlushingIfNeed(this);
  long file_size = utils::get_file_size(fet_file_path_.c_str());
  if (file_size % vector_byte_size_ != 0) {
    LOG(ERROR) << "file_size % vector_byte_size_ != 0, path=" << fet_file_path_;
    return FORMAT_ERR;
  }
  long disk_vector_num = file_size / vector_byte_size_;
  LOG(INFO) << "disk_vector_num=" << disk_vector_num << ", vec_num=" << vec_num;
  assert(disk_vector_num >= vec_num);
  if (disk_vector_num > vec_num) {
    // release file
    if (fet_fd_ != -1) {
      close(fet_fd_);
    }

    if (vector_file_mapper_) {
      delete vector_file_mapper_;
    }

    long trunc_size = (long)vec_num * vector_byte_size_;
    if (truncate(fet_file_path_.c_str(), trunc_size)) {
      LOG(ERROR) << "truncate feature file=" << fet_file_path_ << " to "
                 << trunc_size << ", error:" << strerror(errno);
      return IO_ERR;
    }

    fet_fd_ =
        open(fet_file_path_.c_str(), O_WRONLY | O_APPEND | O_CREAT, 00664);
    if (fet_fd_ == -1) {
      LOG(ERROR) << "open file error:" << strerror(errno);
      return -1;
    }

    vector_file_mapper_ = new VectorFileMapper(fet_file_path_, 1000 * 1000 * 10,
                                               dimension, data_size_);
    if (vector_file_mapper_->Init()) {
      LOG(ERROR) << "vector file mapper map error";
      return INTERNAL_ERR;
    }
    disk_vector_num = vec_num;
  }

  nflushed_ = disk_vector_num;
  last_nflushed_ = nflushed_;
  LOG(INFO) << "load vectors success, nflushed=" << nflushed_;
  StartFlushingIfNeed(this);
  return SUCC;
}

void FreeFileMapper(VectorFileMapper *file_mapper) { delete file_mapper; }

int MmapRawVector::Extend() {
  VectorFileMapper *old_file_mapper = vector_file_mapper_;
  VectorFileMapper *new_mapper =
      new VectorFileMapper(fet_file_path_, max_size_ * 2,
                           meta_info_->Dimension(), meta_info_->DataSize());
  if (new_mapper->Init()) {
    LOG(ERROR) << "extend file mapper, init error, max size=" << max_size_ * 2;
    return INTERNAL_ERR;
  }
  vector_file_mapper_ = new_mapper;
  max_size_ *= 2;
  LOG(INFO) << "extend file mapper sucess, max_size=" << max_size_;
  // delay free old mapper
  std::function<void(VectorFileMapper *)> func_free =
      std::bind(&FreeFileMapper, std::placeholders::_1);
  utils::AsyncWait(1000, func_free, old_file_mapper);
  return SUCC;
}

int MmapRawVector::AddToStore(uint8_t *v, int len) {
  if ((long)meta_info_->Size() >= max_size_ && Extend()) {
    LOG(ERROR) << "extend error";
    return INTERNAL_ERR;
  }
  return vector_buffer_queue_->Push(v, len, -1);
}

int MmapRawVector::GetVectorHeader(int start, int n, ScopeVectors &vecs,
                                   std::vector<int> &lens) {
  if (start + n > (int)meta_info_->Size()) return PARAM_ERR;
  Until(start + n);
  vecs.Add(
      vector_file_mapper_->GetVectors() + (uint64_t)start * vector_byte_size_,
      false);
  lens.push_back(n);
  return SUCC;
}

int MmapRawVector::UpdateToStore(int vid, uint8_t *v, int len) {
  LOG(ERROR) << "MMap doesn't support update!";
  return -1;
};

int MmapRawVector::GetVector(long vid, const uint8_t *&vec,
                             bool &deletable) const {
  if (vid >= (long)meta_info_->Size() || vid < 0) return -1;
  Until((int)vid + 1);
  vec = vector_file_mapper_->GetVector(vid);
  deletable = false;
  return SUCC;
}

}  // namespace tig_gamma
