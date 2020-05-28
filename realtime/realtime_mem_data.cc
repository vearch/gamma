/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "realtime_mem_data.h"
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "bitmap.h"
#include "log.h"
#include "utils.h"

namespace tig_gamma {
namespace realtime {

RTInvertBucketData::RTInvertBucketData(RTInvertBucketData *other) {
  idx_array_ = other->idx_array_;
  retrieve_idx_pos_ = other->retrieve_idx_pos_;
  cur_bucket_keys_ = other->cur_bucket_keys_;
  codes_array_ = other->codes_array_;
  dump_latest_pos_ = other->dump_latest_pos_;
  vid_mgr_ = other->vid_mgr_;
  docids_bitmap_ = other->docids_bitmap_;
  vid_bucket_no_pos_ = other->vid_bucket_no_pos_;
  deleted_nums_ = other->deleted_nums_;
  compacted_num_ = other->compacted_num_;
  buckets_num_ = other->buckets_num_;
}

RTInvertBucketData::RTInvertBucketData(VIDMgr *vid_mgr,
                                       const char *docids_bitmap) {
  idx_array_ = nullptr;
  retrieve_idx_pos_ = nullptr;
  cur_bucket_keys_ = nullptr;
  codes_array_ = nullptr;
  dump_latest_pos_ = nullptr;
  vid_mgr_ = vid_mgr;
  docids_bitmap_ = docids_bitmap;
  vid_bucket_no_pos_ = nullptr;
  deleted_nums_ = nullptr;
  compacted_num_ = 0;
  buckets_num_ = 0;
}

RTInvertBucketData::~RTInvertBucketData() {}

bool RTInvertBucketData::Init(const size_t &buckets_num,
                              const size_t &bucket_keys,
                              const size_t &code_bytes_per_vec,
                              std::atomic<long> &total_mem_bytes,
                              long max_vec_size) {
  idx_array_ = new (std::nothrow) long *[buckets_num];
  codes_array_ = new (std::nothrow) uint8_t *[buckets_num];
  cur_bucket_keys_ = new (std::nothrow) int[buckets_num];
  deleted_nums_ = new (std::nothrow) std::atomic<int>[buckets_num];
  if (idx_array_ == nullptr || codes_array_ == nullptr ||
      cur_bucket_keys_ == nullptr || deleted_nums_ == nullptr)
    return false;
  for (size_t i = 0; i < buckets_num; i++) {
    idx_array_[i] = new (std::nothrow) long[bucket_keys];
    codes_array_[i] =
        new (std::nothrow) uint8_t[bucket_keys * code_bytes_per_vec];
    if (idx_array_[i] == nullptr || codes_array_[i] == nullptr) return false;
    cur_bucket_keys_[i] = bucket_keys;
    deleted_nums_[i] = 0;
  }
  vid_bucket_no_pos_ = new std::atomic<long>[max_vec_size];
  for (int i = 0; i < max_vec_size; i++) vid_bucket_no_pos_[i] = -1;

  total_mem_bytes += buckets_num * bucket_keys * sizeof(long);
  total_mem_bytes +=
      buckets_num * bucket_keys * code_bytes_per_vec * sizeof(uint8_t);
  total_mem_bytes += buckets_num * sizeof(int);

  retrieve_idx_pos_ = new (std::nothrow) int[buckets_num];
  if (retrieve_idx_pos_ == nullptr) return false;
  memset(retrieve_idx_pos_, 0, buckets_num * sizeof(int));
  dump_latest_pos_ = new (std::nothrow) int[buckets_num];
  if (dump_latest_pos_ == nullptr) return false;
  memset(dump_latest_pos_, 0, buckets_num * sizeof(int));
  total_mem_bytes += buckets_num * sizeof(int) * 2;
  buckets_num_ = buckets_num;

  LOG(INFO) << "===init total_mem_bytes is " << total_mem_bytes << "===";
  return true;
}

bool RTInvertBucketData::CompactBucket(const size_t &bucket_no,
                                       const size_t &code_bytes_per_vec) {
  long *old_idx_array = idx_array_[bucket_no];
  uint8_t *old_codes_array = codes_array_[bucket_no];
  int old_keys = cur_bucket_keys_[bucket_no];
  int old_pos = retrieve_idx_pos_[bucket_no];

  long *idx_array = (long *)malloc(sizeof(long) * old_keys);
  uint8_t *codes_array = (uint8_t *)malloc(code_bytes_per_vec * old_keys);
  int pos = 0;
  long *idx_batch_header = nullptr;
  uint8_t *code_batch_header = nullptr;
  int batch_num = 0;
  int new_pos = 0;
  for (int i = 0; i <= old_pos; i++) {
    if (i == old_pos || old_idx_array[i] & kDelIdxMask ||
        bitmap::test(docids_bitmap_,
                     vid_mgr_->VID2DocID(old_idx_array[i] & kRecoverIdxMask))) {
      if (batch_num > 0) {
        memcpy((void *)(idx_array + pos), (void *)idx_batch_header,
               sizeof(long) * batch_num);
        memcpy((void *)(codes_array + pos * code_bytes_per_vec),
               (void *)code_batch_header,
               (size_t)code_bytes_per_vec * batch_num);
        pos += batch_num;
      }

      batch_num = 0;
      idx_batch_header = nullptr;
      code_batch_header = nullptr;
      continue;
    }
    if (!batch_num) {
      idx_batch_header = old_idx_array + i;
      code_batch_header = old_codes_array + i * code_bytes_per_vec;
    }
    vid_bucket_no_pos_[old_idx_array[i]] = bucket_no << 32 | new_pos;
    new_pos++;
    batch_num++;
  }
  idx_array_[bucket_no] = idx_array;
  codes_array_[bucket_no] = codes_array;
  retrieve_idx_pos_[bucket_no] = pos;
  deleted_nums_[bucket_no] = 0;

  compacted_num_ += old_pos - pos;
#ifdef DEBUG
  LOG(INFO) << "compact bucket=" << bucket_no
            << " success! current codes num=" << pos
            << " compacted num=" << old_pos - pos
            << ", total compacted num=" << compacted_num_;
#endif
  return true;
}

bool RTInvertBucketData::ExtendBucketMem(const size_t &bucket_no,
                                         const size_t &code_bytes_per_vec,
                                         std::atomic<long> &total_mem_bytes) {
  int extend_size = cur_bucket_keys_[bucket_no] * 2;

  uint8_t *extend_code_bytes_array =
      new (std::nothrow) uint8_t[extend_size * code_bytes_per_vec];
  if (extend_code_bytes_array == nullptr) {
    LOG(ERROR) << "memory extend_code_bytes_array alloc error!";
    return false;
  }
  memcpy((void *)extend_code_bytes_array, (void *)codes_array_[bucket_no],
         sizeof(uint8_t) * cur_bucket_keys_[bucket_no] * code_bytes_per_vec);
  codes_array_[bucket_no] = extend_code_bytes_array;
  total_mem_bytes += extend_size * code_bytes_per_vec * sizeof(uint8_t);

  long *extend_idx_array = new (std::nothrow) long[extend_size];
  if (extend_idx_array == nullptr) {
    LOG(ERROR) << "memory extend_idx_array alloc error!";
    return false;
  }
  memcpy((void *)extend_idx_array, (void *)idx_array_[bucket_no],
         sizeof(long) * cur_bucket_keys_[bucket_no]);
  idx_array_[bucket_no] = extend_idx_array;
  total_mem_bytes += extend_size * sizeof(long);

  cur_bucket_keys_[bucket_no] = extend_size;

  return true;
}

int RTInvertBucketData::GetCurDumpPos(const size_t &bucket_no, int max_vid,
                                      int &dump_start_pos, int &size) {
  int start_pos = dump_latest_pos_[bucket_no];
  int end_pos = retrieve_idx_pos_[bucket_no];
  if (end_pos == 0) {
    LOG(ERROR) << "bucket no=" << bucket_no << "has no data to dump";
    return -1;
  }
  if (start_pos > end_pos) {
    LOG(ERROR) << "the latest dumping pos exceed the max retrieval pos";
    return -1;
  }
  while ((long)max_vid < idx_array_[bucket_no][--end_pos])
    ;
  if (start_pos > end_pos) {
    return -2;
  }
  dump_start_pos = start_pos;
  size = end_pos - start_pos + 1;
  dump_latest_pos_[bucket_no] += size;
  return 0;
}

void RTInvertBucketData::Delete(int vid) {
  long bucket_no_pos = vid_bucket_no_pos_[vid];
  if (bucket_no_pos == -1) return;  // do nothing
  int bucket_no = bucket_no_pos >> 32;
  // only increase bucket's deleted counter
  deleted_nums_[bucket_no]++;
}

RealTimeMemData::RealTimeMemData(size_t buckets_num, long max_vec_size,
                                 VIDMgr *vid_mgr, const char *docids_bitmap,
                                 size_t bucket_keys, size_t bucket_keys_limit,
                                 size_t code_bytes_per_vec)
    : buckets_num_(buckets_num),
      bucket_keys_(bucket_keys),
      bucket_keys_limit_(bucket_keys_limit),
      code_bytes_per_vec_(code_bytes_per_vec),
      max_vec_size_(max_vec_size),
      vid_mgr_(vid_mgr),
      docids_bitmap_(docids_bitmap) {
  cur_invert_ptr_ = nullptr;
  extend_invert_ptr_ = nullptr;
  total_mem_bytes_ = 0;
}

RealTimeMemData::~RealTimeMemData() {
  if (cur_invert_ptr_) {
    for (size_t i = 0; i < buckets_num_; i++) {
      if (cur_invert_ptr_->idx_array_)
        CHECK_DELETE_ARRAY(cur_invert_ptr_->idx_array_[i]);
      if (cur_invert_ptr_->codes_array_)
        CHECK_DELETE_ARRAY(cur_invert_ptr_->codes_array_[i]);
    }
    CHECK_DELETE_ARRAY(cur_invert_ptr_->idx_array_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->retrieve_idx_pos_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->cur_bucket_keys_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->codes_array_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->dump_latest_pos_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->vid_bucket_no_pos_);
    CHECK_DELETE_ARRAY(cur_invert_ptr_->deleted_nums_);
  }
  CHECK_DELETE(cur_invert_ptr_);
  CHECK_DELETE(extend_invert_ptr_);
}

bool RealTimeMemData::Init() {
  CHECK_DELETE(cur_invert_ptr_);
  cur_invert_ptr_ =
      new (std::nothrow) RTInvertBucketData(vid_mgr_, docids_bitmap_);
  return cur_invert_ptr_ &&
         cur_invert_ptr_->Init(buckets_num_, bucket_keys_, code_bytes_per_vec_,
                               total_mem_bytes_, max_vec_size_);
}

bool RealTimeMemData::AddKeys(size_t list_no, size_t n, std::vector<long> &keys,
                              std::vector<uint8_t> &keys_codes) {
  if (ExtendBucketIfNeed(list_no, n)) return false;

  if (keys.size() * code_bytes_per_vec_ != keys_codes.size()) {
    LOG(ERROR) << "number of key and key codes not match!";
    return false;
  }

  int retrive_pos = cur_invert_ptr_->retrieve_idx_pos_[list_no];
  // copy new added idx to idx buffer

  if (nullptr == cur_invert_ptr_->idx_array_[list_no]) {
    LOG(ERROR) << "-------idx_array is nullptr!--------";
  }
  memcpy((void *)(cur_invert_ptr_->idx_array_[list_no] + retrive_pos),
         (void *)(keys.data()), sizeof(long) * keys.size());

  // copy new added codes to codes buffer
  memcpy((void *)(cur_invert_ptr_->codes_array_[list_no] +
                  retrive_pos * code_bytes_per_vec_),
         (void *)(keys_codes.data()), sizeof(uint8_t) * keys_codes.size());

  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i] >= max_vec_size_) {
      return false;
    }
    cur_invert_ptr_->vid_bucket_no_pos_[keys[i]] = list_no << 32 | retrive_pos;
    retrive_pos++;
    if (bitmap::test(cur_invert_ptr_->docids_bitmap_,
                     cur_invert_ptr_->vid_mgr_->VID2DocID(keys[i]))) {
      cur_invert_ptr_->Delete(keys[i]);
    }
  }

  // atomic switch retriving pos of list_no
  cur_invert_ptr_->retrieve_idx_pos_[list_no] = retrive_pos;

  return true;
}

int RealTimeMemData::Update(int bucket_no, int vid,
                            std::vector<uint8_t> &codes) {
  long bucket_no_pos = cur_invert_ptr_->vid_bucket_no_pos_[vid];
  if (bucket_no_pos == -1) return 0;  // do nothing
  int old_bucket_no = bucket_no_pos >> 32;
  int old_pos = bucket_no_pos & 0xffffffff;
  assert(code_bytes_per_vec_ == codes.size());
  if (old_bucket_no == bucket_no) {
    uint8_t *codes_array = cur_invert_ptr_->codes_array_[old_bucket_no];
    memcpy(codes_array + old_pos * code_bytes_per_vec_, codes.data(),
           codes.size() * sizeof(uint8_t));
    return 0;
  }

  // mark deleted
  cur_invert_ptr_->idx_array_[old_bucket_no][old_pos] |= kDelIdxMask;
  cur_invert_ptr_->deleted_nums_[old_bucket_no]++;
  std::vector<long> keys;
  keys.push_back(vid);

  return AddKeys(bucket_no, 1, keys, codes);
}

int RealTimeMemData::Delete(int *vids, int n) {
  for (int i = 0; i < n; i++) {
    RTInvertBucketData *invert_ptr = cur_invert_ptr_;
    invert_ptr->Delete(vids[i]);
  }
  return 0;
}

void RealTimeMemData::FreeOldData(long *idx, uint8_t *codes,
                                  RTInvertBucketData *invert, long size) {
  if (idx) {
    delete idx;
    idx = nullptr;
  }
  if (codes) {
    delete codes;
    codes = nullptr;
  }
  if (invert) {
    delete invert;
    invert = nullptr;
  }
  total_mem_bytes_ -= size;
}

int RealTimeMemData::CompactIfNeed() {
  long last_compacted_num = cur_invert_ptr_->compacted_num_;
  for (int i = 0; i < (int)buckets_num_; i++) {
    if (Compactable(i)) {
      if (!CompactBucket(i)) {
        LOG(ERROR) << "compact bucket=" << i << " error!";
        return -2;
      }
    }
  }
  if (cur_invert_ptr_->compacted_num_ > last_compacted_num) {
    LOG(INFO) << "Compaction happened, compacted num="
              << cur_invert_ptr_->compacted_num_ - last_compacted_num
              << ", last compacted num=" << last_compacted_num
              << ", current compacted num=" << cur_invert_ptr_->compacted_num_;
  }
  return 0;
}

bool RealTimeMemData::Compactable(int bucket_no) {
  return (float)cur_invert_ptr_->deleted_nums_[bucket_no] /
             cur_invert_ptr_->retrieve_idx_pos_[bucket_no] >=
         0.3f;
}

bool RealTimeMemData::CompactBucket(int bucket_no) {
  return AdjustBucketMem(bucket_no, 1);
}

int RealTimeMemData::ExtendBucketIfNeed(int bucket_no, size_t keys_size) {
  if (cur_invert_ptr_->retrieve_idx_pos_[bucket_no] + (int)keys_size <=
      cur_invert_ptr_->cur_bucket_keys_[bucket_no]) {
    return 0;
  } else {  // can not add new keys any more
    if (cur_invert_ptr_->cur_bucket_keys_[bucket_no] * 2 >=
        (int)bucket_keys_limit_) {
      LOG(WARNING) << "exceed the max bucket keys [" << bucket_keys_limit_
                   << "], not extend memory any more!"
                   << " keys_size [" << keys_size << "] "
                   << "bucket_no [" << bucket_no << "]"
                   << " cur_invert_ptr_->cur_bucket_keys_[bucket_no] ["
                   << cur_invert_ptr_->cur_bucket_keys_[bucket_no] << "]";
      return -1;
    } else {
#if 0  // memory limit
        utils::MEM_PACK *p = utils::get_memoccupy();
        if (p->used_rate > 80) {
          LOG(WARNING)
              << "System memory used [" << p->used_rate
              << "]%, cannot add doc, keys_size [" << keys_size << "]"
              << "bucket_no [" << bucket_no << "]"
              << " cur_invert_ptr_->cur_bucket_keys_[bucket_no] ["
              << cur_invert_ptr_->cur_bucket_keys_[bucket_no] << "]";
          free(p);
          return false;
        } else {
          LOG(INFO) << "System memory used [" << p->used_rate << "]%";
          free(p);
        }
#endif
      if (!ExtendBucketMem(bucket_no)) {
        return -2;
      }
    }
  }
  return 0;
}

bool RealTimeMemData::ExtendBucketMem(const size_t &bucket_no) {
  return AdjustBucketMem(bucket_no, 0);
}

bool RealTimeMemData::AdjustBucketMem(const size_t &bucket_no, int type) {
  extend_invert_ptr_ = new (std::nothrow) RTInvertBucketData(cur_invert_ptr_);
  if (!extend_invert_ptr_) {
    LOG(ERROR) << "memory extend_invert_ptr_ alloc error!";
    return false;
  }

  long *old_idx_array = cur_invert_ptr_->idx_array_[bucket_no];
  uint8_t *old_codes_array = cur_invert_ptr_->codes_array_[bucket_no];
  int old_keys = cur_invert_ptr_->cur_bucket_keys_[bucket_no];
  long free_size = old_keys * sizeof(long) +
                   old_keys * code_bytes_per_vec_ * sizeof(uint8_t);

  if (type == 0) {  // extend bucket
    // WARNING:
    // the above idx_array_ and codes_array_ pointer would be changed by
    // extendBucketMem()
    if (!extend_invert_ptr_->ExtendBucketMem(bucket_no, code_bytes_per_vec_,
                                             total_mem_bytes_)) {
      LOG(ERROR) << "extendBucketMem error!";
      return false;
    }
  } else {  // compact bucket
    if (!extend_invert_ptr_->CompactBucket(bucket_no, code_bytes_per_vec_)) {
      LOG(ERROR) << "compact error!";
      return false;
    }
    free_size = 0;
  }

  RTInvertBucketData *old_invert_ptr = cur_invert_ptr_;
  cur_invert_ptr_ = extend_invert_ptr_;

  std::function<void(long *, uint8_t *, RTInvertBucketData *, long)> func_free =
      std::bind(&RealTimeMemData::FreeOldData, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4);

  utils::AsyncWait(1000, func_free, old_idx_array, old_codes_array,
                   old_invert_ptr, free_size);

  old_idx_array = nullptr;
  old_codes_array = nullptr;
  old_invert_ptr = nullptr;
  extend_invert_ptr_ = nullptr;

  return true;
}

bool RealTimeMemData::GetIvtList(const size_t &bucket_no, long *&ivt_list,
                                 uint8_t *&ivt_codes_list) {
  ivt_list = cur_invert_ptr_->idx_array_[bucket_no];
  ivt_codes_list = (uint8_t *)(cur_invert_ptr_->codes_array_[bucket_no]);

  return true;
}

int RealTimeMemData::RetrieveCodes(
    int *vids, size_t vid_size,
    std::vector<std::vector<const uint8_t *>> &bucket_codes,
    std::vector<std::vector<long>> &bucket_vids) {
  bucket_codes.resize(buckets_num_);
  bucket_vids.resize(buckets_num_);
  for (size_t i = 0; i < buckets_num_; i++) {
    bucket_codes[i].reserve(vid_size / buckets_num_);
    bucket_vids[i].reserve(vid_size / buckets_num_);
  }

  for (size_t i = 0; i < vid_size; i++) {
    if (cur_invert_ptr_->vid_bucket_no_pos_[vids[i]] != -1) {
      int bucket_no = cur_invert_ptr_->vid_bucket_no_pos_[vids[i]] >> 32;
      int pos = cur_invert_ptr_->vid_bucket_no_pos_[vids[i]] & 0xffffffff;
      bucket_codes[bucket_no].push_back(
          cur_invert_ptr_->codes_array_[bucket_no] + pos * code_bytes_per_vec_);
      bucket_vids[bucket_no].push_back(vids[i]);
    }
  }

  return 0;
}

int RealTimeMemData::RetrieveCodes(
    int **vids_list, size_t vids_list_size,
    std::vector<std::vector<const uint8_t *>> &bucket_codes,
    std::vector<std::vector<long>> &bucket_vids) {
  bucket_codes.resize(buckets_num_);
  bucket_vids.resize(buckets_num_);
  for (size_t i = 0; i < buckets_num_; i++) {
    bucket_codes[i].reserve(vids_list_size / buckets_num_);
    bucket_vids[i].reserve(vids_list_size / buckets_num_);
  }

  for (size_t i = 0; i < vids_list_size; i++) {
    for (int j = 1; j <= vids_list[i][0]; j++) {
      int vid = vids_list[i][j];
      if (cur_invert_ptr_->vid_bucket_no_pos_[vid] != -1) {
        int bucket_no = cur_invert_ptr_->vid_bucket_no_pos_[vid] >> 32;
        int pos = cur_invert_ptr_->vid_bucket_no_pos_[vid] & 0xffffffff;
        bucket_codes[bucket_no].push_back(
            cur_invert_ptr_->codes_array_[bucket_no] +
            pos * code_bytes_per_vec_);
        bucket_vids[bucket_no].push_back(vid);
      }
    }
  }

  return 0;
}

int RealTimeMemData::Dump(const std::string &dir, const std::string &vec_name,
                          int max_vid) {
  int buckets[buckets_num_];
  long *ids[buckets_num_];
  uint8_t *codes[buckets_num_];
  LOG(INFO) << "dump max vector id=" << max_vid;

  int ids_count = 0;
  int real_dump_min_vid = INT_MAX, real_dump_max_vid = -1;
  for (size_t i = 0; i < buckets_num_; i++) {
    int start_pos = -1;
    int size = 0;
    if (cur_invert_ptr_->GetCurDumpPos(i, max_vid, start_pos, size) == 0) {
      ids[i] = cur_invert_ptr_->idx_array_[i] + start_pos;
      codes[i] =
          cur_invert_ptr_->codes_array_[i] + (start_pos * code_bytes_per_vec_);
      int bucket_min_vid = cur_invert_ptr_->idx_array_[i][start_pos];
      int bucket_max_vid = cur_invert_ptr_->idx_array_[i][start_pos + size - 1];
#ifdef DEBUG
      LOG(INFO) << "dump bucket no=" << i << ", min vid=" << bucket_min_vid
                << ", max vid=" << bucket_max_vid << ", size=" << size
                << ", dir=" << dir
                << ", dump_latest_pos_=" << cur_invert_ptr_->dump_latest_pos_[i]
                << ", vids=" << utils::join(ids[i], size, ',');
#endif
      if (real_dump_min_vid > bucket_min_vid) {
        real_dump_min_vid = bucket_min_vid;
      }
      if (real_dump_max_vid < bucket_max_vid) {
        real_dump_max_vid = bucket_max_vid;
      }
    }
    buckets[i] = size;
    ids_count += size;
  }

  if (ids_count > 0) {
    std::string dump_file = dir + "/" + vec_name + ".index";
    FILE *fp = fopen(dump_file.c_str(), "wb");

    fwrite((void *)&ids_count, sizeof(int), 1, fp);
    fwrite((void *)&real_dump_min_vid, sizeof(int), 1, fp);
    fwrite((void *)&real_dump_max_vid, sizeof(int), 1, fp);
    fwrite((void *)&buckets_num_, sizeof(int), 1, fp);
    fwrite((void *)buckets, sizeof(int), buckets_num_, fp);
    for (size_t i = 0; i < buckets_num_; i++) {
      fwrite((void *)ids[i], sizeof(long), buckets[i], fp);
      fwrite((void *)codes[i], sizeof(uint8_t),
             buckets[i] * code_bytes_per_vec_, fp);
    }
    fclose(fp);
    LOG(INFO) << "ids_count=" << ids_count
              << ", real_dump_min_vid=" << real_dump_min_vid
              << ", real_dump_max_vid=" << real_dump_max_vid
              << ", buckets_num_=" << buckets_num_
              << ", buckets=" << utils::join<int>(buckets, buckets_num_, ',');
  }
  return ids_count;
}

int RealTimeMemData::Load(const std::vector<std::string> &index_dirs,
                          const std::string &vec_name) {
  size_t indexes_num = index_dirs.size();
  int ids_count[indexes_num], min_vids[indexes_num], max_vids[indexes_num];
  int bucket_ids[indexes_num][buckets_num_];
  FILE *fp_array[indexes_num];

  int total_bucket_ids[buckets_num_], total_ids = 0;
  memset((void *)total_bucket_ids, 0, buckets_num_ * sizeof(int));
  for (size_t i = 0; i < indexes_num; i++) {
    std::string index_file = index_dirs[i] + "/" + vec_name + ".index";
    if (access(index_file.c_str(), F_OK) != 0) {
      fp_array[i] = nullptr;
      continue;
    }
    fp_array[i] = fopen(index_file.c_str(), "rb");
    fread((void *)(ids_count + i), sizeof(int), 1, fp_array[i]);
    fread((void *)(min_vids + i), sizeof(int), 1, fp_array[i]);
    fread((void *)(max_vids + i), sizeof(int), 1, fp_array[i]);
    int buckets_num = 0;
    fread((void *)&buckets_num, sizeof(int), 1, fp_array[i]);
    if ((size_t)buckets_num != buckets_num_) {
      LOG(ERROR) << "buckets_num must be " << buckets_num_;
      continue;
    }
    if (ids_count[i] == 0 || min_vids[i] == INT_MAX || max_vids[i] == -1) {
      LOG(INFO) << " no data in the bucket " << i
                << " of real time index dumped";
      continue;
    }
    if (i > 0 && max_vids[i - 1] != -1 && min_vids[i] != 0 &&
        (min_vids[i] != (max_vids[i - 1] + 1))) {
      std::string last_index_file =
          index_dirs[i - 1] + "/" + vec_name + ".index";
      LOG(ERROR) << "the file " << index_file
                 << " missing some vectors after the file " << last_index_file;
    }

    fread((void *)bucket_ids[i], sizeof(int), buckets_num_, fp_array[i]);
    for (size_t j = 0; j < buckets_num_; j++) {
      total_bucket_ids[j] += bucket_ids[i][j];
      total_ids += bucket_ids[i][j];
    }
  }
  long *load_bucket_ids[buckets_num_];
  uint8_t *load_bucket_codes[buckets_num_];
  total_mem_bytes_ = buckets_num_ * sizeof(int) * 3;
  for (size_t i = 0; i < buckets_num_; i++) {
    size_t total_keys = total_bucket_ids[i] * 2;
    // if (total_keys > bucket_keys_) {
    //   total_keys = bucket_keys_;
    // }
    load_bucket_ids[i] = new long[total_keys];
    total_mem_bytes_ += total_keys * sizeof(long);
    load_bucket_codes[i] = new uint8_t[total_keys * code_bytes_per_vec_];
    total_mem_bytes_ += total_keys * code_bytes_per_vec_ * sizeof(uint8_t);
    cur_invert_ptr_->cur_bucket_keys_[i] = total_keys;
    cur_invert_ptr_->retrieve_idx_pos_[i] = total_bucket_ids[i];
  }

  int ids_load_offset_list[buckets_num_], codes_load_offset_list[buckets_num_];
  memset(ids_load_offset_list, 0, sizeof(ids_load_offset_list));
  memset(codes_load_offset_list, 0, sizeof(codes_load_offset_list));
  for (size_t i = 0; i < indexes_num; i++) {
    if (!fp_array[i]) {
      continue;
    }
    for (size_t j = 0; j < buckets_num_; j++) {
      fread((void *)(load_bucket_ids[j] + ids_load_offset_list[j]),
            sizeof(long), bucket_ids[i][j], fp_array[i]);
#ifdef DEBUG
      long min_vid = load_bucket_ids[j][ids_load_offset_list[j]];
      long max_vid =
          load_bucket_ids[j][ids_load_offset_list[j] + bucket_ids[i][j] - 1];
      LOG(INFO) << "index id=" << i << ", bucket no=" << j
                << ", min vid=" << min_vid << ", max vid=" << max_vid
                << ", size=" << bucket_ids[i][j];
#endif
      ids_load_offset_list[j] += bucket_ids[i][j];
      int codes_count = bucket_ids[i][j] * code_bytes_per_vec_;
      fread((void *)(load_bucket_codes[j] + codes_load_offset_list[j]),
            sizeof(uint8_t), codes_count, fp_array[i]);
      codes_load_offset_list[j] += codes_count;
    }
    fclose(fp_array[i]);
  }

  /* switch the ids and codes memory pointer */
  for (size_t i = 0; i < buckets_num_; i++) {
    delete[] cur_invert_ptr_->idx_array_[i];
    cur_invert_ptr_->idx_array_[i] = load_bucket_ids[i];
    delete[] cur_invert_ptr_->codes_array_[i];
    cur_invert_ptr_->codes_array_[i] = load_bucket_codes[i];
    cur_invert_ptr_->dump_latest_pos_[i] = total_bucket_ids[i];
#ifdef DEBUG
    LOG(INFO) << "bucket id=" << i
              << ", dump_latest_pos_=" << cur_invert_ptr_->dump_latest_pos_[i];
#endif
  }

  int bucket_size = 0;
  long vid = -1;
  for (size_t bucket_id = 0; bucket_id < buckets_num_; bucket_id++) {
    bucket_size = cur_invert_ptr_->retrieve_idx_pos_[bucket_id];
    for (int retrive_pos = 0; retrive_pos < bucket_size; retrive_pos++) {
      vid = cur_invert_ptr_->idx_array_[bucket_id][retrive_pos];
      if (vid >= max_vec_size_ || vid < 0) {
        LOG(INFO) << "invalid vid=" << vid
                  << ", max vector size=" << max_vec_size_;
        return -1;
      }
      cur_invert_ptr_->vid_bucket_no_pos_[vid] = bucket_id << 32 | retrive_pos;
    }
  }
  return total_ids;
}

void RealTimeMemData::PrintBucketSize() {
  std::vector<std::pair<size_t, int>> buckets;

  for (size_t bucket_id = 0; bucket_id < buckets_num_; ++bucket_id) {
    int bucket_size = cur_invert_ptr_->retrieve_idx_pos_[bucket_id];
    buckets.push_back(std::make_pair(bucket_id, bucket_size));
  }

  std::sort(
      buckets.begin(), buckets.end(),
      [](const std::pair<size_t, int> &a, const std::pair<size_t, int> &b) {
        return (a.second > b.second);
      });

  std::stringstream ss;
  ss << "Bucket (id, size): ";
  for (const auto &bucket : buckets) {
    ss << "(" << bucket.first << ", " << bucket.second << ") ";
  }
  LOG(INFO) << ss.str();
}
}  // namespace realtime

}  // namespace tig_gamma
