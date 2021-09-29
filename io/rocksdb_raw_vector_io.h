#ifdef WITH_ROCKSDB

#ifndef ROCKSDB_RAW_VECTOR_IO_H_
#define ROCKSDB_RAW_VECTOR_IO_H_

#pragma once

#include <string>
#include "raw_vector_io.h"
#include "vector/rocksdb_raw_vector.h"

namespace tig_gamma {

struct RocksDBRawVectorIO : public RawVectorIO {
  RocksDBRawVector *raw_vector;

  RocksDBRawVectorIO(RocksDBRawVector *raw_vector_) : raw_vector(raw_vector_) {}
  ~RocksDBRawVectorIO() {}
  int Init() override { return 0; };
  int Dump(int start_vid, int end_vid) override { return 0; };
  int GetDiskVecNum(int &vec_num) override;
  int Load(int vec_num) override;
  int Update(int vid) override { return 0; };
};

}  // namespace tig_gamma

#endif

#endif // WITH_ROCKSDB
