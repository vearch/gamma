#ifndef RAW_VECTOR_COMMON_H_
#define RAW_VECTOR_COMMON_H_

#include <string.h>
#include "utils.h"

const static int MAX_VECTOR_NUM_PER_DOC = 10;
const static int MAX_CACHE_SIZE = 1024 * 1024;  // M bytes, it is equal to 1T

template <typename DataType>
struct ScopeVector {
  const DataType *ptr_;
  bool deletable_;

  explicit ScopeVector(const DataType *ptr = nullptr) : ptr_(ptr) {}
  void Set(const DataType *ptr_in, bool deletable = true) {
    ptr_ = ptr_in;
    deletable_ = deletable;
  }
  const DataType *Get() { return ptr_; }
  ~ScopeVector() {
    if (deletable_ && ptr_) delete[] ptr_;
  }
};

template <typename DataType>
struct ScopeVectors {
  const DataType **ptr_;
  int size_;
  bool *deletable_;

  explicit ScopeVectors(int size) : size_(size) {
    ptr_ = new const DataType *[size_];
    deletable_ = new bool[size_];
  }
  void Set(int idx, const DataType *ptr_in, bool deletable = true) {
    ptr_[idx] = ptr_in;
    deletable_[idx] = deletable;
  }
  const DataType **Get() { return ptr_; }
  const DataType *Get(int idx) { return ptr_[idx]; }
  ~ScopeVectors() {
    for (int i = 0; i < size_; i++) {
      if (deletable_[i] && ptr_[i]) delete[] ptr_[i];
    }
    delete[] deletable_;
    delete[] ptr_;
  }
};

struct VIDMgr {
  std::vector<int> vid2docid_;    // vector id to doc id
  std::vector<int *> docid2vid_;  // doc id to vector id list
  bool multi_vids_;

  VIDMgr(bool multi_vids) : multi_vids_(multi_vids) {}

  ~VIDMgr() {
    if (multi_vids_) {
      for (size_t i = 0; i < docid2vid_.size(); i++) {
        if (docid2vid_[i] != nullptr) {
          delete[] docid2vid_[i];
          docid2vid_[i] = nullptr;
        }
      }
    }
  }

  int Init(int max_vector_size, long &total_mem_bytes) {
    if (multi_vids_) {
      vid2docid_.resize(max_vector_size, -1);
      total_mem_bytes += max_vector_size * sizeof(int);
      docid2vid_.resize(max_vector_size, nullptr);
      total_mem_bytes += max_vector_size * sizeof(docid2vid_[0]);
    }
    return 0;
  }

  int Add(int vid, int docid) {
    // add to vid2docid_ and docid2vid_
    if (multi_vids_) {
      vid2docid_[vid] = docid;
      if (docid2vid_[docid] == nullptr) {
        docid2vid_[docid] =
            utils::NewArray<int>(MAX_VECTOR_NUM_PER_DOC + 1, "init_vid_list");
        // total_mem_bytes += (MAX_VECTOR_NUM_PER_DOC + 1) * sizeof(int);
        docid2vid_[docid][0] = 1;
        docid2vid_[docid][1] = vid;
      } else {
        int *vid_list = docid2vid_[docid];
        if (vid_list[0] + 1 > MAX_VECTOR_NUM_PER_DOC) {
          return -1;
        }
        vid_list[vid_list[0]] = vid;
        vid_list[0]++;
      }
    }
    return 0;
  }

  inline int VID2DocID(int vid) {
    if (!multi_vids_) return vid;
    return vid2docid_[vid];
  }

  inline void DocID2VID(int docid, std::vector<int> &vids) {
    if (!multi_vids_) {
      vids.resize(1);
      vids[0] = docid;
      return;
    }
    int *vid_list = docid2vid_[docid];
    vids.resize(vid_list[0]);
    memcpy((void *)vids.data(), (void *)(vid_list + 1), vid_list[0] * sizeof(int));
    return;
  }

  inline int GetFirstVID(int docid) {
    if (!multi_vids_) {
      return docid;
    }
    int *vid_list = docid2vid_[docid];
    if (vid_list[0] <= 0) return -1;
    return vid_list[1];
  }

  inline int GetLastVID(int docid) {
    if (!multi_vids_) {
      return docid;
    }
    int *vid_list = docid2vid_[docid];
    if (vid_list[0] <= 0) return -1;
    return vid_list[vid_list[0]];
  }
};

#endif  // RAW_VECTOR_COMMON_H_
