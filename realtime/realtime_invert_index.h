/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef REALTIME_INVERT_INDEX_H_
#define REALTIME_INVERT_INDEX_H_

#include <stdlib.h>
#include <map>
#include <vector>
#include "bitmap.h"
#include "faiss/Index.h"
#include "faiss/IndexIVF.h"

#include "realtime_mem_data.h"

namespace tig_gamma {
namespace realtime {

struct RTInvertIndex {
 public:
  // bucket_keys should not be larger than bucket_keys_limit
  RTInvertIndex(faiss::Index *index, long max_vec_size,
                const char *docids_bitmap, int *vid2docid,
                size_t bucket_keys = 10000, size_t bucket_keys_limit = 1000000);

  ~RTInvertIndex();

  bool Init();

  /*  @param n : count of added keys
   *  @param keys : added key arrays
   *  @param keys_codes : added key code arrays*/
  bool AddKeys(std::map<int, std::vector<long>> &new_keys,
               std::map<int, std::vector<uint8_t>> &new_codes);

  int Update(int bucket_no, int vid, std::vector<uint8_t> &codes);

  inline faiss::IndexIVF *GetIndexIVF() { return _index_ivf; }

  bool GetIvtList(const size_t &bucket_no, long *&ivt_list, size_t &ivt_size,
                  uint8_t *&ivt_codes_list);

  long GetTotalMemBytes() {
    return _cur_ptr ? _cur_ptr->GetTotalMemBytes() : 0;
  }

  int RetrieveCodes(int *vids, size_t vid_size,
                    std::vector<std::vector<const uint8_t *>> &bucket_codes,
                    std::vector<std::vector<long>> &bucket_vids);

  int RetrieveCodes(int **vids_list, size_t vids_list_size,
                    std::vector<std::vector<const uint8_t *>> &bucket_codes,
                    std::vector<std::vector<long>> &bucket_vids);

  int Dump(const std::string &file_path, const std::string &vec_name,
           int max_vid);
  int Load(const std::vector<std::string> &index_dirs,
           const std::string &vec_name);

  void PrintBucketSize();
  int CompactBucket(int bucket_no);
  int RetrieveBucketId(int vid);

  bool Compactable(int deleted_doc_num);

 private:
  size_t _bucket_keys;
  size_t _bucket_keys_limit;
  long _max_vec_size;
  faiss::IndexIVF *_index_ivf;
  const char *docids_bitmap_;
  int *vid2docid_;

  RealTimeMemData *_cur_ptr;
};

}  // namespace realtime

}  // namespace tig_gamma

#endif
