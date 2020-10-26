#include "mmap_raw_vector_io.h"
#include "error_code.h"
#include "io_common.h"

namespace tig_gamma {

int MmapRawVectorIO::Init() { return 0; }

int MmapRawVectorIO::Dump(int start_vid, int end_vid) {
  for (int i = start_vid / raw_vector->segment_size_;
       i < end_vid / raw_vector->segment_size_; i++) {
    int ret = raw_vector->file_mappers_[i]->Sync();
    if (ret) return ret;
  }
  return 0;
}

int MmapRawVectorIO::Load(int vec_num) {
  int seg_num = vec_num / raw_vector->segment_size_ + 1;
  int offset = vec_num % raw_vector->segment_size_;
  for (int i = 1; i < seg_num; ++i) {
    int ret = raw_vector->Extend();
    if (ret) {
      LOG(ERROR) << "load extend error, i=" << i << ", ret=" << ret;
      return ret;
    }
  }
  assert(raw_vector->nsegment_ == seg_num);
  raw_vector->file_mappers_[seg_num - 1]->SetCurrIdx(offset);
  raw_vector->MetaInfo()->size_ = vec_num;
  LOG(INFO) << "mmap load success! vec num=" << vec_num;
  return 0;
}

int MmapRawVectorIO::Update(int vid) {
  return 0;  // do nothing
}

}  // namespace tig_gamma
