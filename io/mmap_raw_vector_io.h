#ifndef MMAP_RAW_VECTOR_IO_H_
#define MMAP_RAW_VECTOR_IO_H_

#pragma once

#include <string>
#include "mmap_raw_vector.h"
#include "raw_vector_io.h"

namespace tig_gamma {

struct MmapRawVectorIO : public RawVectorIO {
  MmapRawVector *raw_vector;

  MmapRawVectorIO(MmapRawVector *raw_vector_) : raw_vector(raw_vector_) {}
  ~MmapRawVectorIO() {}

  int Init() override;
  // [start_vid, end_vid)
  int Dump(int start_vid, int end_vid) override;
  int Load(int vec_num) override;
  int Update(int vid) override;
};

}  // namespace tig_gamma

#endif
