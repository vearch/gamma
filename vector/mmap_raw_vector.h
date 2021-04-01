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
#include "storage_manager.h"

namespace tig_gamma {

class MmapRawVectorIO;

class MmapRawVector : public RawVector {
 public:
  MmapRawVector(VectorMetaInfo *meta_info, const std::string &root_path,
                const StoreParams &store_params, const char *docids_bitmap);
  ~MmapRawVector();
  int InitStore() override;
  int AddToStore(uint8_t *v, int len) override;
  int GetVectorHeader(int start, int n, ScopeVectors &vecs,
                      std::vector<int> &lens) override;
  int UpdateToStore(int vid, uint8_t *v, int len) override;

  int AlterCacheSize(uint32_t cache_size) override;

  int GetCacheSize(uint32_t &cache_size) override;

 protected:
  int GetVector(long vid, const uint8_t *&vec, bool &deletable) const override;

 private:
  int Extend();
  std::string NextSegmentFilePath();

 private:
  friend MmapRawVectorIO;
  StorageManager *storage_mgr_;
};

}  // namespace tig_gamma

#endif  // MMAP_RAW_VECTOR_H_
