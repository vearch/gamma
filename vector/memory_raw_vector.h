/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef MEMORY_RAW_VECTOR_H_
#define MEMORY_RAW_VECTOR_H_

#include <string>

#include "raw_vector.h"
#include "rocksdb_wrapper.h"

namespace tig_gamma {
static const int kMaxSegments = 1000;

class MemoryRawVector : public RawVector {
 public:
  MemoryRawVector(VectorMetaInfo *meta_info, const std::string &root_path,
                  const StoreParams &store_params, const char *docids_bitmap);

  ~MemoryRawVector();

  int InitStore() override;

  int AddToStore(uint8_t *v, int len) override;

  int GetVectorHeader(int start, int n, ScopeVectors &vecs,
                      std::vector<int> &lens) override;

  int UpdateToStore(int vid, uint8_t *v, int len) override;

  int LoadVectors(int vec_num) override;

 protected:
  int GetVector(long vid, const uint8_t *&vec, bool &deleteable) const override;

 private:
  int ExtendSegments();
  int AddToMem(uint8_t *v, int len);

  uint8_t **segments_;
  int nsegments_;
  int segment_size_;
  uint8_t *current_segment_;
  int curr_idx_in_seg_;
#ifdef WITH_ROCKSDB
  RocksDBWrapper rdb_;
#endif
};

}  // namespace tig_gamma

#endif
