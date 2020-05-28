/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef _REALTIME_MEM_DATA_H_
#define _REALTIME_MEM_DATA_H_

#include <stdint.h>
#include <stdlib.h>
#include <atomic>
#include <string>
#include <vector>
#include "raw_vector_common.h"

namespace tig_gamma {

namespace realtime {

const static long kDelIdxMask = (long)1 << 63;     // 0x8000000000000000
const static long kRecoverIdxMask = ~kDelIdxMask;  // 0x7fffffffffffffff

struct RTInvertBucketData {
  RTInvertBucketData(RTInvertBucketData *other);
  RTInvertBucketData(VIDMgr *vid_mgr, const char *docids_bitmap);

  bool Init(const size_t &buckets_num, const size_t &bucket_keys,
            const size_t &code_bytes_per_vec,
            std::atomic<long> &total_mem_bytes, long max_vec_size);
  ~RTInvertBucketData();

  bool ExtendBucketMem(const size_t &bucket_no,
                       const size_t &code_bytes_per_vec,
                       std::atomic<long> &total_mem_bytes);

  int GetCurDumpPos(const size_t &bucket_no, int max_vid, int &dump_start_pos,
                    int &size);

  bool CompactBucket(const size_t &bucket_no, const size_t &code_bytes_per_vec);

  void Delete(int vid);

  long **idx_array_;
  int *retrieve_idx_pos_;  // total nb of realtime added indexed vectors
  int *cur_bucket_keys_;
  uint8_t **codes_array_;
  int *dump_latest_pos_;
  VIDMgr *vid_mgr_;
  const char *docids_bitmap_;
  std::atomic<long> *vid_bucket_no_pos_;
  std::atomic<int> *deleted_nums_;
  long compacted_num_;
  size_t buckets_num_;
};

struct RealTimeMemData {
 public:
  RealTimeMemData(size_t buckets_num, long max_vec_size, VIDMgr *vid_mgr,
                  const char *docids_bitmap, size_t bucket_keys = 500,
                  size_t bucket_keys_limit = 1000000,
                  size_t code_bytes_per_vec = 512 * sizeof(float));
  ~RealTimeMemData();

  bool Init();

  bool AddKeys(size_t list_no, size_t n, std::vector<long> &keys,
               std::vector<uint8_t> &keys_codes);

  int Update(int bucket_no, int vid, std::vector<uint8_t> &codes);

  void FreeOldData(long *idx, uint8_t *codes, RTInvertBucketData *invert,
                   long size);
  int ExtendBucketIfNeed(int bucket_no, size_t keys_size);
  bool ExtendBucketMem(const size_t &bucket_no);
  bool AdjustBucketMem(const size_t &bucket_no, int type);
  bool GetIvtList(const size_t &bucket_no, long *&ivt_list,
                  uint8_t *&ivt_codes_list);

  long GetTotalMemBytes() { return total_mem_bytes_; }

  int RetrieveCodes(int *vids, size_t vid_size,
                    std::vector<std::vector<const uint8_t *>> &bucket_codes,
                    std::vector<std::vector<long>> &bucket_vids);

  int RetrieveCodes(int **vids_list, size_t vids_list_size,
                    std::vector<std::vector<const uint8_t *>> &bucket_codes,
                    std::vector<std::vector<long>> &bucket_vids);

  int Dump(const std::string &dir, const std::string &vec_name, int max_vid);
  int Load(const std::vector<std::string> &index_dirs,
           const std::string &vec_name);

  void PrintBucketSize();

  int CompactIfNeed();
  bool Compactable(int bucket_no);
  bool CompactBucket(int bucket_no);
  int Delete(int *vids, int n);

  RTInvertBucketData *cur_invert_ptr_;
  RTInvertBucketData *extend_invert_ptr_;

  size_t buckets_num_;  // count of buckets
  size_t bucket_keys_;  // max bucket keys
  size_t bucket_keys_limit_;

  size_t code_bytes_per_vec_;
  std::atomic<long> total_mem_bytes_;

  long max_vec_size_;
  VIDMgr *vid_mgr_;
  const char *docids_bitmap_;
};

}  // namespace realtime

}  // namespace tig_gamma

#endif
