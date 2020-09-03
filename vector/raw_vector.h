/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef RAW_VECTOR_H_
#define RAW_VECTOR_H_

#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "api_data/gamma_doc.h"
#include "concurrentqueue/concurrentqueue.h"
#include "log.h"
#include "raw_vector_common.h"
#include "retrieval_model.h"
#include "utils.h"

namespace tig_gamma {

class RawVectorIO;
struct StoreParams;

static const int kInitSize = 1000 * 1000;

class RawVector : public VectorReader {
 public:
  RawVector(VectorMetaInfo *meta_info, const std::string &root_path, 
	    const char *docids_bitmap, const StoreParams &store_params);

  virtual ~RawVector();

  /** initialize resource
   *
   * @return 0 if successed
   */
  int Init(bool has_source, bool multi_vids);

  /** get the header of vectors, so it can access vecotors through the
   * header if dimension is known
   *
   * @param start start vector id(include)
   * @param n number of vectors
   * @param vec[out] vector header address
   * @param m[out] the real number of vectors(0 < m <= n)
   * @return success: 0
   */
  virtual int GetVectorHeader(int start, int n, ScopeVectors &vec,
                              std::vector<int> &lens) = 0;

  /** dump vectors and sources to disk file
   *
   * @param path  the disk directory path
   * @return 0 if successed
   */
  int Dump(const std::string &path, int dump_docid, int max_docid);

  /** load vectors and sources from disk file
   *
   * @param path  the disk directory path
   * @return 0 if successed
   */
  int Load(const std::vector<std::string> &path, int doc_num);

  /** get vector by id
   *
   * @param id  vector id
   * @return    vector if successed, null if failed
   */
  int GetVector(long vid, ScopeVector &vec) const;

  /** get vectors by vecotor id list
   *
   * @param ids_list  vector id list
   * @param resultss  (output) vectors, Warning: the vectors must be destroyed
   * by Destroy()
   * @return 0 if successed
   */
  virtual int Gets(const std::vector<int64_t> &vids, ScopeVectors &vecs) const;

  /** get source of one vector, source is a string, for example the image url of
   *  vector
   *
   * @param vid   vector id
   * @param str   (output) the pointer of source string
   * @param len   (output) the len of source string
   * @return 0    if successed
   */
  int GetSource(int vid, char *&str, int &len);

  /** add one vector field
   *
   * @param docid doc id, one doc may has multiple vectors
   * @param field vector field, it contains vector(uint8_t array) and
   * source(string)
   * @return 0 if successed
   */
  int Add(int docid, struct Field &field);

  int Update(int docid, struct Field &field);

  virtual size_t GetStoreMemUsage() { return 0; }

  long GetTotalMemBytes() {
    GetStoreMemUsage();
    return total_mem_bytes_;
  };

  int GetVectorNum() const { return meta_info_->Size(); };

	int IndexedVectorNum() const { return indexed_vector_num_; };

	void SetIndexedVectorNum(int indexed_vector_num) { 
		indexed_vector_num_ = indexed_vector_num; 
	};

  /** add vector to the specific implementation of RawVector(memory or disk)
   *it is called by next common function Add()
   */
  virtual int AddToStore(uint8_t *v, int len) = 0;

  virtual int UpdateToStore(int vid, uint8_t *v, int len) = 0;

  VIDMgr *VidMgr() const { return vid_mgr_; }
  const char *Bitmap() { return docids_bitmap_; }
  moodycamel::ConcurrentQueue<int> *UpdatedVids() { return updated_vids_; }
  moodycamel::ConcurrentQueue<int> *updated_vids_;

 protected:
  /** get vector by id
   *
   * @param id vector id
   * @return vector if successed, null if failed
   */
  virtual int GetVector(long vid, const uint8_t *&vec,
                        bool &deletable) const = 0;

  virtual int DumpVectors(int dump_vid, int n) { return 0; }

  virtual int LoadVectors(int vec_num) { return 0; }

  virtual int InitStore() = 0;

  int Compress(uint8_t *v, ScopeVector &svec);
  int Decompress(uint8_t *cmpr_v, int n, uint8_t *&vec, bool &deletable) const;

 protected:
  friend RawVectorIO;
  std::string root_path_;
  int vector_byte_size_;
  int data_size_;

  long total_mem_bytes_;              // total used memory bytes
  char *str_mem_ptr_;                 // source memory
  std::vector<long> source_mem_pos_;  // position of each source
  bool has_source_;
  std::string desc_;  // description of this raw vector

  StoreParams *store_params_;
#ifdef WITH_ZFP
  ZFPCompressor zfp_compressor_;
#endif
	int indexed_vector_num_;
  const char *docids_bitmap_;
  VIDMgr *vid_mgr_;
};

class RawVectorIO {
 public:
  RawVectorIO(RawVector *raw_vector);

  ~RawVectorIO();

  int Init();

  int Dump(int start, int n);

  int Load(int doc_num);

 private:
  RawVector *raw_vector_;
  int docid_fd_;
  int src_fd_;
  int src_pos_fd_;
};

class AsyncFlusher {
 public:
  AsyncFlusher(std::string name);

  ~AsyncFlusher();

  void Start();

  void Stop();

  void Until(int nexpect) const;

 protected:
  static void Handler(AsyncFlusher *flusher);

  int Flush();

  virtual int FlushOnce() = 0;

 protected:
  std::string name_;
  std::thread *runner_;
  bool stopped_;
  long nflushed_;
  long last_nflushed_;
  int interval_;
};

void StartFlushingIfNeed(RawVector *vec);

void StopFlushingIfNeed(RawVector *vec);

struct StoreParams {
  long cache_size_;  // bytes
  int segment_size_;
  bool compress_;

  StoreParams() {
    cache_size_ = 1024 * 1024 * 1024;  // 1G
    segment_size_ = 1000000;           // 1M
    compress_ = false;
  }

  StoreParams(const StoreParams &other) {
    this->cache_size_ = other.cache_size_;
    this->segment_size_ = other.segment_size_;
    this->compress_ = other.compress_;
  }

  int Parse(const char *str);

  std::string ToString() {
    std::stringstream ss;
    ss << "{cache size=" << cache_size_ << ", segment size=" << segment_size_
       << ", compress=" << compress_ << "}";
    return ss.str();
  }

  std::string ToJson() {
    std::stringstream ss;
    ss << "{";
    ss << "\"cache_size\":" << cache_size_ << ",";
    ss << "\"segment_size\":" << segment_size_ << ",";
    std::string cmps = compress_ ? "true" : "false";
    ss << "\"compress\":" << cmps;
    ss << "}";
    return ss.str();
  }
};

}  // namespace tig_gamma
#endif /* RAW_VECTOR_H_ */
