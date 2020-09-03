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

class MmapRawVector : public RawVector, public AsyncFlusher {
 public:
  MmapRawVector(VectorMetaInfo *meta_info, const std::string &root_path,
                const StoreParams &store_params, const char *docids_bitmap);
  ~MmapRawVector();
  int InitStore() override;
  int AddToStore(uint8_t *v, int len) override;
  int GetVectorHeader(int start, int n, ScopeVectors &vecs,
                      std::vector<int> &lens) override;
  // currently it doesn't support update
  int UpdateToStore(int vid, uint8_t *v, int len) override;

 protected:
  int FlushOnce() override;
  int GetVector(long vid, const uint8_t *&vec, bool &deletable) const override;
  int DumpVectors(int dump_vid, int max_vid) override;
  int LoadVectors(int vec_num) override;

 private:
  int Extend();

 private:
  VectorBufferQueue *vector_buffer_queue_;
  VectorFileMapper *vector_file_mapper_;
  int max_buffer_size_;
  int buffer_chunk_num_;
  int flush_batch_size_;
  int flush_write_retry_;
  uint8_t *flush_batch_vectors_;
  std::string fet_file_path_;
  int fet_fd_;
  long max_size_;
};

}  // namespace tig_gamma

#endif  // MMAP_RAW_VECTOR_H_
