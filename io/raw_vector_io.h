#ifndef RAW_VECTOR_IO_H_
#define RAW_VECTOR_IO_H_

#pragma once

#include <string>

namespace tig_gamma {

struct RawVectorIO {
  virtual ~RawVectorIO(){};
  
  virtual int Init() { return 0; }
  // [start_vid, end_vid)
  virtual int Dump(int start_vid, int end_vid) = 0;
  virtual int Load(int vec_num) = 0;
  virtual int Update(int vid) = 0;
};

}  // namespace tig_gamma
#endif
