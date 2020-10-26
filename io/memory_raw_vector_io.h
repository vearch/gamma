#ifdef WITH_ROCKSDB

#ifndef MEMORY_RAW_VECTOR_IO_H_
#define MEMORY_RAW_VECTOR_IO_H_

#pragma once

#include <string>
#include "async_flush.h"
#include "memory_raw_vector.h"
#include "raw_vector_io.h"

namespace tig_gamma {

struct MemoryRawVectorIO : public RawVectorIO, public AsyncFlusher {
  MemoryRawVector *raw_vector;
  RocksDBWrapper rdb;

  MemoryRawVectorIO(MemoryRawVector *raw_vector_)
      : AsyncFlusher(raw_vector_->MetaInfo()->Name()),
        raw_vector(raw_vector_) {}
  ~MemoryRawVectorIO() {}
  int Init() override;
  int Dump(int start_vid, int end_vid) override;
  int Load(int vec_num) override;
  int Update(int vid) override;

  int FlushOnce() override;

  int Put(int vid);
};

}  // namespace tig_gamma

#endif

#endif // WITH_ROCKSDB
