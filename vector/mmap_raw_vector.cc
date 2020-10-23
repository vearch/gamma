/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "mmap_raw_vector.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <exception>
#include "error_code.h"
#include "log.h"
#include "utils.h"

using namespace std;

namespace tig_gamma {

MmapRawVector::MmapRawVector(VectorMetaInfo *meta_info, const string &root_path,
                             const StoreParams &store_params,
                             const char *docids_bitmap)
    : RawVector(meta_info, root_path, docids_bitmap, store_params) {
  vector_byte_size_ = meta_info_->DataSize() * meta_info->Dimension();
  nsegment_ = 0;
  segment_size_ = store_params.segment_size;
}

MmapRawVector::~MmapRawVector() {
  for (int i = 0; i < file_mappers_.size(); i++) {
    CHECK_DELETE(file_mappers_[i]);
  }
}

string MmapRawVector::NextSegmentFilePath() {
  char buf[7];
  snprintf(buf, 7, "%06d", nsegment_);
  string vec_dir = root_path_ + "/" + meta_info_->Name();
  string file_path = vec_dir + "/vector-" + buf;
  return file_path;
}

int MmapRawVector::InitStore() {
  std::string vec_dir = root_path_ + "/" + meta_info_->Name();
  if (utils::make_dir(vec_dir.c_str())) {
    LOG(ERROR) << "mkdir error, path=" << vec_dir;
    return IO_ERR;
  }
  file_mappers_.resize(kMaxSegments, nullptr);
  int ret = Extend();
  if (ret) return ret;

  LOG(INFO) << "Init success! vector byte size=" << vector_byte_size_
            << ", segment size=" << segment_size_;
  return 0;
}

int MmapRawVector::Extend() {
  VectorFileMapper *file_mapper = new VectorFileMapper(
      NextSegmentFilePath(), segment_size_, vector_byte_size_);
  int ret = file_mapper->Init();
  if (ret) {
    LOG(ERROR) << "extend file mapper error, ret=" << ret;
    return ret;
  }
  file_mappers_[nsegment_++] = file_mapper;
  return 0;
}

int MmapRawVector::AddToStore(uint8_t *v, int len) {
  if (file_mappers_[nsegment_ - 1]->IsFull() && Extend()) {
    LOG(ERROR) << "extend error";
    return INTERNAL_ERR;
  }
  return file_mappers_[nsegment_ - 1]->Add(v, len);
}

int MmapRawVector::GetVectorHeader(int start, int n, ScopeVectors &vecs,
                                   std::vector<int> &lens) {
  if (start + n > meta_info_->Size()) return PARAM_ERR;
  while (n) {
    int offset = start % segment_size_;
    vecs.Add(file_mappers_[start / segment_size_]->GetVector(offset), false);
    int len = segment_size_ - offset;
    if (len > n) len = n;
    lens.push_back(len);
    start += len;
    n -= len;
  }
  return 0;
}

int MmapRawVector::UpdateToStore(int vid, uint8_t *v, int len) {
  if (vid >= (long)meta_info_->Size() || vid < 0 || len != vector_byte_size_) {
    return PARAM_ERR;
  }
  return file_mappers_[vid / segment_size_]->Update(vid % segment_size_,  v, len);
};

int MmapRawVector::GetVector(long vid, const uint8_t *&vec,
                             bool &deletable) const {
  if (vid >= meta_info_->Size() || vid < 0) return -1;
  vec = file_mappers_[vid / segment_size_]->GetVector(vid % segment_size_);
  deletable = false;
  return 0;
}

}  // namespace tig_gamma
