/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "memory_raw_vector.h"
#include "mmap_raw_vector.h"
#include "raw_vector.h"

#ifdef WITH_ROCKSDB
#include "rocksdb_raw_vector.h"
#endif  // WITH_ROCKSDB

#include <string>
#include "gamma_common_data.h"

#include "gamma_common_data.h"

namespace tig_gamma {
class RawVectorFactory {
 public:
  static RawVector *Create(VectorMetaInfo *meta_info, VectorStorageType type,
                           const std::string &root_path,
                           const std::string &store_param,
                           const char *docids_bitmap) {
    StoreParams store_params;
    // TODO remove store_param != ""
    if (store_param != "" && store_params.Parse(store_param.c_str())) {
      return nullptr;
    }

    LOG(INFO) << "Store parameters [" << store_params.ToString() << "]";
    switch (type) {
      case VectorStorageType::MemoryOnly:
        return (RawVector *)new MemoryRawVector(meta_info, root_path,
                                                store_params, docids_bitmap);
      case VectorStorageType::Mmap:
        if (store_params.compress_) {
          LOG(ERROR) << "mmap unsupport compress";
          return nullptr;
        }
        return (RawVector *)new MmapRawVector(meta_info, root_path,
                                              store_params, docids_bitmap);
#ifdef WITH_ROCKSDB
      case VectorStorageType::RocksDB:
        return (RawVector *)new RocksDBRawVector(meta_info, root_path,
                                                 store_params, docids_bitmap);
#endif  // WITH_ROCKSDB
      default:
        LOG(ERROR) << "invalid raw feature type:" << static_cast<int>(type);
        return nullptr;
    }
  }
};

}  // namespace tig_gamma
