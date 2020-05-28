/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "raw_vector.h"
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "log.h"
#include "utils.h"

using namespace std;

namespace tig_gamma {

template <typename DataType>
RawVectorIO<DataType>::RawVectorIO(RawVector<DataType> *raw_vector) {
  raw_vector_ = raw_vector;
  docid_fd_ = -1;
  src_fd_ = -1;
  src_pos_fd_ = -1;
}

template <typename DataType>
RawVectorIO<DataType>::~RawVectorIO() {
  if (docid_fd_ != -1) close(docid_fd_);
  if (src_fd_ != -1) close(src_fd_);
  if (src_pos_fd_ != -1) close(src_pos_fd_);
}

template <typename DataType>
int RawVectorIO<DataType>::Init() {
  string docid_file_path =
      raw_vector_->root_path_ + "/" + raw_vector_->vector_name_ + ".docid";
  string src_file_path =
      raw_vector_->root_path_ + "/" + raw_vector_->vector_name_ + ".src";
  string src_pos_file_path =
      raw_vector_->root_path_ + "/" + raw_vector_->vector_name_ + ".src.pos";
  docid_fd_ = open(docid_file_path.c_str(), O_RDWR | O_APPEND | O_CREAT, 00664);
  src_fd_ = open(src_file_path.c_str(), O_RDWR | O_APPEND | O_CREAT, 00664);
  src_pos_fd_ =
      open(src_pos_file_path.c_str(), O_RDWR | O_APPEND | O_CREAT, 00664);
  if (docid_fd_ == -1 || src_fd_ == -1 || src_pos_fd_ == -1) {
    LOG(ERROR) << "open file error:" << strerror(errno);
    return -1;
  }
  return 0;
}
template <typename DataType>
int RawVectorIO<DataType>::Dump(int start, int n) {
  if (raw_vector_->has_source_) {
    char *str_mem_ptr = raw_vector_->str_mem_ptr_;
    long *source_mem_pos = raw_vector_->source_mem_pos_.data();

    // dump source
    if (str_mem_ptr) {
      write(src_fd_, (void *)(str_mem_ptr + source_mem_pos[start]),
            source_mem_pos[start + n] - source_mem_pos[start]);
    }

    // dump source position
    if (start == 0) {
      write(src_pos_fd_, (void *)(source_mem_pos + start),
            (n + 1) * sizeof(long));
    } else {
      write(src_pos_fd_, (void *)(source_mem_pos + start + 1),
            n * sizeof(long));
    }
  }

  if (raw_vector_->vid_mgr_->multi_vids_) {
    int *vid2docid = raw_vector_->vid_mgr_->vid2docid_.data();
    write(docid_fd_, (void *)(vid2docid + start), n * sizeof(int));
  }

#ifdef DEBUG
  LOG(INFO) << "io dump,  start=" << start << ", n=" << n;
#endif

  return 0;
}

template <typename DataType>
int RawVectorIO<DataType>::Load(int doc_num) {
  if (doc_num == 0) {
    if (ftruncate(docid_fd_, 0)) {
      LOG(ERROR) << "truncate docid file error:" << strerror(errno);
      return -1;
    }
    if (ftruncate(src_pos_fd_, 0)) {
      LOG(ERROR) << "truncate source position file error:" << strerror(errno);
      return -1;
    }
    if (ftruncate(src_fd_, 0)) {
      LOG(ERROR) << "truncate source file error:" << strerror(errno);
      return -1;
    }
    return 0;
  }
  int n = 0;
  if (raw_vector_->vid_mgr_->multi_vids_) {
    string docid_file_path =
        raw_vector_->root_path_ + "/" + raw_vector_->vector_name_ + ".docid";
    long docid_file_size = utils::get_file_size(docid_file_path.c_str());
    if (docid_file_size <= 0 || docid_file_size % sizeof(int) != 0) {
      LOG(ERROR) << "invalid docid file size=" << docid_file_size;
      return -1;
    }
    int num = docid_file_size / sizeof(int);
    read(docid_fd_, (void *)raw_vector_->vid_mgr_->vid2docid_.data(),
         num * sizeof(int));
    // create docid2vid_ from vid2docid_
    int vid = 0;
    for (; vid < num; vid++) {
      int docid = raw_vector_->vid_mgr_->vid2docid_[vid];
      if (docid == -1) {
        continue;
      }
      if (docid >= doc_num) {
        break;
      }
      raw_vector_->vid_mgr_->Add(vid, docid);
    }
    n = vid;
    // set [n, num) to be -1
    for (int i = n; i < num; i++) {
      raw_vector_->vid_mgr_->vid2docid_[i] = -1;
    }

    // truncate docid file to vid_num length
    if (ftruncate(docid_fd_, n * sizeof(int))) {
      LOG(ERROR) << "truncate docid file error:" << strerror(errno);
      return -1;
    }
  }

  if (raw_vector_->has_source_) {
    read(src_pos_fd_, (void *)raw_vector_->source_mem_pos_.data(),
         (n + 1) * sizeof(long));
    if (raw_vector_->source_mem_pos_[n] > 0) {
      read(src_fd_, (void *)raw_vector_->str_mem_ptr_,
           raw_vector_->source_mem_pos_[n]);
    }

    // truncate str file to vid_num length
    if (ftruncate(src_pos_fd_, (n + 1) * sizeof(long))) {
      LOG(ERROR) << "truncate source position file error:" << strerror(errno);
      return -1;
    }
    if (ftruncate(src_fd_, raw_vector_->source_mem_pos_[n])) {
      LOG(ERROR) << "truncate source file error:" << strerror(errno);
      return -1;
    }
  }
  if (raw_vector_->vid_mgr_->multi_vids_)
    return n;
  else
    return doc_num;
}

template <typename DataType>
RawVector<DataType>::RawVector(const string &name, int dimension,
                               int max_vector_size, const string &root_path)
    : vector_name_(name),
      dimension_(dimension),
      max_vector_size_(max_vector_size),
      root_path_(root_path),
      ntotal_(0),
      total_mem_bytes_(0) {}

template <typename DataType>
RawVector<DataType>::~RawVector() {
  CHECK_DELETE_ARRAY(str_mem_ptr_);
  CHECK_DELETE(updated_vids_);
  CHECK_DELETE(vid_mgr_);
}

template <typename DataType>
int RawVector<DataType>::Init(bool has_source, bool multi_vids) {
  // source
  str_mem_ptr_ = nullptr;
  if (has_source) {
    uint64_t len = (uint64_t)max_vector_size_ * 100;
    str_mem_ptr_ = new (std::nothrow) char[len];
    total_mem_bytes_ += len;
    source_mem_pos_.resize(max_vector_size_ + 1, 0);
    total_mem_bytes_ += max_vector_size_ * sizeof(long);
  }
  has_source_ = has_source;

  // vid2docid
  vid_mgr_ = new VIDMgr(multi_vids);
  vid_mgr_->Init(max_vector_size_, total_mem_bytes_);

  vector_byte_size_ = dimension_ * sizeof(DataType);
  updated_vids_ = new moodycamel::ConcurrentQueue<int>();
  int ret = InitStore();
  if (ret) return ret;
  LOG(INFO) << "raw vector init success! name=" << vector_name_
            << ", has source=" << has_source << ", multi_vids=" << multi_vids;
  return 0;
}

template <typename DataType>
int RawVector<DataType>::GetVector(long vid, ScopeVector<DataType> &vec) {
  return GetVector(vid, vec.ptr_, vec.deletable_);
}

template <typename DataType>
int RawVector<DataType>::Dump(const std::string &path, int dump_docid,
                              int max_docid) {
  LOG(INFO) << "dump_docid=" << dump_docid << ", max_docid=" << max_docid;
  int start = vid_mgr_->GetFirstVID(dump_docid);
  int end = vid_mgr_->GetLastVID(max_docid);
  int n = end - start + 1;
  RawVectorIO<DataType> *raw_vector_io = new RawVectorIO<DataType>(this);
  if (raw_vector_io->Init()) return -1;
  raw_vector_io->Dump(start, n);
  delete raw_vector_io;
  return DumpVectors(start, n);
};

template <typename DataType>
int RawVector<DataType>::Load(const std::vector<std::string> &path,
                              int doc_num) {
  RawVectorIO<DataType> *raw_vector_io = new RawVectorIO<DataType>(this);
  if (raw_vector_io->Init()) return -1;
  int num = raw_vector_io->Load(doc_num);
  delete raw_vector_io;
  assert(num >= 0);
  if (LoadVectors(num)) {
    LOG(ERROR) << "load vectors error";
    return -2;
  }
  ntotal_ = num;
  return 0;
}

template <typename DataType>
int RawVector<DataType>::Gets(int k, long *ids_list,
                              ScopeVectors<DataType> &vecs) const {
  bool deletable;
  for (int i = 0; i < k; i++) {
    const DataType *vec = nullptr;
    deletable = false;
    GetVector(ids_list[i], vec, deletable);
    vecs.Set(i, vec, deletable);
  }
  return 0;
}

template <typename DataType>
int RawVector<DataType>::GetSource(int vid, char *&str, int &len) {
  if (vid < 0 || vid >= ntotal_) return -1;
  if (!has_source_) {
    str = nullptr;
    len = 0;
    return 0;
  }
  len = source_mem_pos_[vid + 1] - source_mem_pos_[vid];
  str = str_mem_ptr_ + source_mem_pos_[vid];
  return 0;
}

template <typename DataType>
int RawVector<DataType>::Add(int docid, Field *&field) {
  if (ntotal_ >= max_vector_size_) {
    return -1;
  }
  if (field->value->len / sizeof(DataType) <= 0) {
    LOG(ERROR) << "Doc [" << docid << "] len " << field->value->len << "]";
    return -1;
  }
  AddToStore((DataType *)field->value->value,
             field->value->len / sizeof(DataType));

  // add to source
  if (has_source_) {
    int len = field->source ? field->source->len : 0;
    if (len > 0) {
      memcpy(str_mem_ptr_ + source_mem_pos_[ntotal_], field->source->value,
             len * sizeof(char));
      source_mem_pos_[ntotal_ + 1] = source_mem_pos_[ntotal_] + len;
    } else {
      source_mem_pos_[ntotal_ + 1] = source_mem_pos_[ntotal_];
    }
  }

  return vid_mgr_->Add(ntotal_++, docid);
}

template <typename DataType>
int RawVector<DataType>::Update(int docid, Field *&field) {
  if (vid_mgr_->multi_vids_ || docid >= ntotal_) return -1;
  int vid = docid;

  if (field->value->len / sizeof(DataType) <= 0) {
    LOG(ERROR) << "Doc [" << docid << "] len " << field->value->len << "]";
    return -1;
  }

  if (UpdateToStore(vid, (DataType *)field->value->value,
                    field->value->len / sizeof(DataType))) {
    LOG(ERROR) << "update to store error, docid=" << docid;
    return -1;
  }
  updated_vids_->enqueue(vid);
  // TODO: update source
  return 0;
}

AsyncFlusher::AsyncFlusher(string name) : name_(name) {
  stopped_ = false;
  last_nflushed_ = nflushed_ = 0;
  interval_ = 100;  // ms
  runner_ = nullptr;
}

AsyncFlusher::~AsyncFlusher() {
  if (runner_) delete runner_;
}

void AsyncFlusher::Start() {
  // TODO: check if it is stopped
  stopped_ = false;
  runner_ = new std::thread(Handler, this);
}

void AsyncFlusher::Stop() {
  stopped_ = true;
  if (runner_) {
    runner_->join();
    delete runner_;
    runner_ = nullptr;
  }
}

void AsyncFlusher::Handler(tig_gamma::AsyncFlusher *flusher) {
  LOG(INFO) << "flusher=" << flusher->name_ << " is started!";
  int ret = flusher->Flush();
  if (ret != 0) {
    LOG(ERROR) << "flusher=" << flusher->name_
               << " exit unexpectedly! ret=" << ret;
  } else {
    LOG(INFO) << "flusher=" << flusher->name_ << " exit successfully!";
  }
}

int AsyncFlusher::Flush() {
  while (!stopped_) {
    int ret = FlushOnce();
    if (ret < 0)
      return ret;
    else
      nflushed_ += ret;
    if (nflushed_ - last_nflushed_ > 100) {
#ifdef DEBUG
      LOG(INFO) << "flushed number=" << nflushed_;
#endif
      last_nflushed_ = nflushed_;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
  }
  return 0;
}

void AsyncFlusher::Until(int nexpect) {
  while (nflushed_ < nexpect) {
    LOG(INFO) << "flusher waiting......, expected num=" << nexpect
              << ", flushed num=" << nflushed_;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int StoreParams::Parse(const char *str) {
  utils::JsonParser jp;
  if (jp.Parse(str)) {
    LOG(ERROR) << "parse store parameters error: " << str;
    return -1;
  }

  double cache_size = 0;
  if (!jp.GetDouble("cache_size", cache_size)) {
    if (cache_size > MAX_CACHE_SIZE || cache_size < 0) {
      LOG(ERROR) << "invalid cache size=" << cache_size << "M"
                 << ", limit size=" << MAX_CACHE_SIZE << "M";
      return -1;
    }
    cache_size_ = (long)cache_size * 1024 * 1024;
  }

  return 0;
}

template class RawVector<float>;
template class RawVector<uint8_t>;

template class RawVectorIO<float>;
template class RawVectorIO<uint8_t>;

}  // namespace tig_gamma
