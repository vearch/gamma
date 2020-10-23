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
#include "rocksdb_raw_vector_io.h"
#endif  // WITH_ROCKSDB

#include <string>
#include "gamma_common_data.h"

#include "memory_raw_vector_io.h"
#include "mmap_raw_vector_io.h"

namespace tig_gamma {

static void Fail(RawVector *raw_vector, RawVectorIO *vio, std::string err_msg) {
  LOG(ERROR) << err_msg;
  if (raw_vector) delete raw_vector;
  if (vio) delete vio;
}

class RawVectorFactory {
 public:
  static RawVector *Create(VectorMetaInfo *meta_info, VectorStorageType type,
                           const std::string &root_path,
                           StoreParams &store_params,
                           const char *docids_bitmap) {
    RawVector *raw_vector = nullptr;
    RawVectorIO *vio = nullptr;
    switch (type) {
      case VectorStorageType::MemoryOnly:
        raw_vector = new MemoryRawVector(meta_info, root_path, store_params,
                                         docids_bitmap);
        store_params.store_type = "rocksdb";
        vio = new MemoryRawVectorIO((MemoryRawVector *)raw_vector);
        break;
      case VectorStorageType::Mmap:
        if (!store_params.compress.IsEmpty()) {
          LOG(ERROR) << "mmap unsupport compress";
          return nullptr;
        }
        raw_vector = new MmapRawVector(meta_info, root_path, store_params,
                                       docids_bitmap);
        store_params.store_type = "file";
        vio = new MmapRawVectorIO((MmapRawVector *)raw_vector);
	break;
#ifdef WITH_ROCKSDB
      case VectorStorageType::RocksDB:
        raw_vector = new RocksDBRawVector(meta_info, root_path, store_params,
                                          docids_bitmap);
        vio = new RocksDBRawVectorIO((RocksDBRawVector *)raw_vector);
        break;
#endif  // WITH_ROCKSDB
      default:
        LOG(ERROR) << "invalid raw feature type:" << static_cast<int>(type);
        return nullptr;
    }
    if (vio && vio->Init()) {
      Fail(raw_vector, vio, "init raw vector io error");
      return nullptr;
    }
    raw_vector->SetIO(vio);
    return raw_vector;
  }
};

/* void StartFlushingIfNeed(RawVector *vec) { */
/*   AsyncFlusher *flusher = dynamic_cast<AsyncFlusher *>(vec->GetIO()); */
/*   if (flusher) { */
/*     flusher->Start(); */
/*     const std::string &name = vec->MetaInfo()->Name(); */
/*     LOG(INFO) << "start flushing, raw vector=" << name; */
/*   } */
/* } */

/* void StopFlushingIfNeed(RawVector *vec) { */
/*   AsyncFlusher *flusher = dynamic_cast<AsyncFlusher *>(vec->GetIO()); */
/*   if (flusher) { */
/*     flusher->Until(vec->GetVectorNum()); */
/*     flusher->Stop(); */
/*     const std::string &name = vec->MetaInfo()->Name(); */
/*     LOG(INFO) << "stop flushing, raw vector=" << name; */
/*   } */
/* } */

}  // namespace tig_gamma
