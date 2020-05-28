/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef MMAP_RAW_VECTOR_H_
#define MMAP_RAW_VECTOR_H_

#include <string>
#include <thread>
#include "raw_vector.h"
#include "vector_buffer_queue.h"
#include "vector_file_mapper.h"

namespace tig_gamma {

static const int kDefaultBufferChunkNum = 1024;

template <typename DataType>
class MmapRawVector : public RawVector<DataType>, public AsyncFlusher {
 public:
  MmapRawVector(const std::string &name, int dimension, int max_vector_size,
                const std::string &root_path, const StoreParams &store_params);
  ~MmapRawVector();
  int InitStore() override;
  int AddToStore(DataType *v, int len) override;
  int GetVectorHeader(int start, int end, ScopeVector<DataType> &vec) override;
  int UpdateToStore(int vid, DataType *v, int len);
  int GetMemoryMode() { return memory_only_; }

 protected:
  int FlushOnce() override;
  int GetVector(long vid, const DataType *&vec, bool &deletable) const override;
  int DumpVectors(int dump_vid, int max_vid);
  int LoadVectors(int vec_num) override;
  int LoadUpdatedVectors();

 private:
  VectorBufferQueue<DataType> *vector_buffer_queue_;
  VectorFileMapper<DataType> *vector_file_mapper_;
  int max_buffer_size_;
  int buffer_chunk_num_;
  int flush_batch_size_;
  int flush_write_retry_;
  int init_vector_num_;
  DataType *flush_batch_vectors_;
  std::string fet_file_path_;
  std::string updated_fet_file_path_;
  int fet_fd_;
  FILE *updated_fet_fp_;
  StoreParams *store_params_;
  int stored_num_;
  bool memory_only_;
};

}  // namespace tig_gamma

#endif  // MMAP_RAW_VECTOR_H_
