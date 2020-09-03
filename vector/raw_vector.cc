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

RawVectorIO::RawVectorIO(RawVector *raw_vector) {
  raw_vector_ = raw_vector;
  docid_fd_ = -1;
  src_fd_ = -1;
  src_pos_fd_ = -1;
}

RawVectorIO::~RawVectorIO() {
  if (docid_fd_ != -1) close(docid_fd_);
  if (src_fd_ != -1) close(src_fd_);
  if (src_pos_fd_ != -1) close(src_pos_fd_);
}

int RawVectorIO::Init() {
  const std::string &vector_name = raw_vector_->MetaInfo()->Name();

  std::string docid_file_path =
      raw_vector_->root_path_ + "/" + vector_name + ".docid";
  std::string src_file_path =
      raw_vector_->root_path_ + "/" + vector_name + ".src";
  std::string src_pos_file_path =
      raw_vector_->root_path_ + "/" + vector_name + ".src.pos";
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

int RawVectorIO::Dump(int start, int n) {
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

  if (raw_vector_->VidMgr()->MultiVids()) {
    int *vid2docid = raw_vector_->VidMgr()->Vid2Docid().data();
    write(docid_fd_, (void *)(vid2docid + start), n * sizeof(int));
  }

#ifdef DEBUG
  LOG(INFO) << "io dump,  start=" << start << ", n=" << n;
#endif

  return 0;
}

int RawVectorIO::Load(int doc_num) {
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
  if (raw_vector_->VidMgr()->MultiVids()) {
    const std::string &vector_name = raw_vector_->MetaInfo()->Name();
    string docid_file_path =
        raw_vector_->root_path_ + "/" + vector_name + ".docid";
    long docid_file_size = utils::get_file_size(docid_file_path.c_str());
    if (docid_file_size <= 0 || docid_file_size % sizeof(int) != 0) {
      LOG(ERROR) << "invalid docid file size=" << docid_file_size;
      return -1;
    }
    int num = docid_file_size / sizeof(int);
    read(docid_fd_, (void *)raw_vector_->VidMgr()->Vid2Docid().data(),
         num * sizeof(int));
    // create docid2vid_ from Vid2Docid()
    int vid = 0;
    for (; vid < num; vid++) {
      int docid = raw_vector_->VidMgr()->Vid2Docid()[vid];
      if (docid == -1) {
        continue;
      }
      if (docid >= doc_num) {
        break;
      }
      raw_vector_->VidMgr()->Add(vid, docid);
    }
    n = vid;
    // set [n, num) to be -1
    for (int i = n; i < num; i++) {
      raw_vector_->VidMgr()->Vid2Docid()[i] = -1;
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
  if (raw_vector_->VidMgr()->MultiVids())
    return n;
  else
    return doc_num;
}

RawVector::RawVector(VectorMetaInfo *meta_info, const string &root_path,
                     const char *docids_bitmap, const StoreParams &store_params)
    : VectorReader(meta_info),
      root_path_(root_path),
      total_mem_bytes_(0),
      indexed_vector_num_(0),
      docids_bitmap_(docids_bitmap) {
  data_size_ = meta_info_->DataSize();
  store_params_ = new StoreParams(store_params);
}

RawVector::~RawVector() {
  CHECK_DELETE_ARRAY(str_mem_ptr_);
  CHECK_DELETE(updated_vids_);
  CHECK_DELETE(vid_mgr_);
  CHECK_DELETE(store_params_);
}

int RawVector::Init(bool has_source, bool multi_vids) {
  desc_ += "raw vector=" + meta_info_->Name() + ", ";
  if (has_source || multi_vids) {
    LOG(ERROR) << "source and multi-vids is unsupported now";
    return -1;
  }
  // source
  str_mem_ptr_ = nullptr;
  if (has_source) {
    uint64_t len = (uint64_t)kInitSize * 100;
    str_mem_ptr_ = new (std::nothrow) char[len];
    source_mem_pos_.resize(kInitSize + 1, 0);
    total_mem_bytes_ += len + kInitSize * sizeof(long);
  }
  has_source_ = has_source;

  // vid2docid
  vid_mgr_ = new VIDMgr(multi_vids);
  vid_mgr_->Init(kInitSize, total_mem_bytes_);

  vector_byte_size_ = meta_info_->Dimension() * data_size_;
  updated_vids_ = new moodycamel::ConcurrentQueue<int>();

#ifdef WITH_ZFP
  if (store_params_->compress_) {
    if (meta_info_->DataType() != VectorValueType::FLOAT) {
      LOG(ERROR) << "data type is not float, compress is unsupported";
      return PARAM_ERR;
    }
    zfp_compressor_.Init(meta_info_->Dimension());
    vector_byte_size_ = zfp_compressor_.ZfpSize();
  }
#endif

  if (InitStore()) return -2;

  LOG(INFO) << "raw vector init success! name=" << meta_info_->Name()
            << ", has source=" << has_source << ", multi_vids=" << multi_vids
            << ", vector_byte_size=" << vector_byte_size_
            << ", dimension=" << meta_info_->Dimension()
            << ", compress=" << store_params_->compress_;
  return 0;
}

int RawVector::GetVector(long vid, ScopeVector &vec) const {
  return GetVector(vid, vec.Get(), vec.Deletable());
}

int RawVector::Dump(const std::string &path, int dump_docid, int max_docid) {
  LOG(INFO) << "dump_docid=" << dump_docid << ", max_docid=" << max_docid;
  int start = vid_mgr_->GetFirstVID(dump_docid);
  int end = vid_mgr_->GetLastVID(max_docid);
  int n = end - start + 1;
  // TODO: dump source and docids
  return DumpVectors(start, n);
};

int RawVector::Load(const std::vector<std::string> &path, int doc_num) {
  // TODO: load source and docids
  int num = doc_num;
  if (LoadVectors(num)) {
    LOG(ERROR) << "load vectors error";
    return -2;
  }
  meta_info_->size_ = num;
  return 0;
}

int RawVector::Gets(const std::vector<int64_t> &vids,
                    ScopeVectors &vecs) const {
  bool deletable;
  for (size_t i = 0; i < vids.size(); i++) {
    const uint8_t *vec = nullptr;
    deletable = false;
    GetVector(vids[i], vec, deletable);
    vecs.Add(vec, deletable);
  }
  return 0;
}

int RawVector::GetSource(int vid, char *&str, int &len) {
  if (vid < 0 || vid >= (int)meta_info_->Size()) return -1;
  if (!has_source_) {
    str = nullptr;
    len = 0;
    return 0;
  }
  len = source_mem_pos_[vid + 1] - source_mem_pos_[vid];
  str = str_mem_ptr_ + source_mem_pos_[vid];
  return 0;
}

int RawVector::Add(int docid, struct Field &field) {
  if (field.value.size() != (size_t)data_size_ * meta_info_->Dimension()) {
    LOG(ERROR) << "Doc [" << docid << "] len " << field.value.size() << "]";
    return -1;
  }
  int ret = AddToStore((uint8_t *)field.value.c_str(), field.value.size());
  if (ret) {
    LOG(ERROR) << "add to store error, docid=" << docid << ", ret=" << ret;
    return -2;
  }

  // add to source
  if (has_source_) {
    size_t size = meta_info_->Size();
    int len = field.source.size();
    if (len > 0) {
      memcpy(str_mem_ptr_ + source_mem_pos_[size], field.source.c_str(),
             len * sizeof(char));
      source_mem_pos_[size + 1] = source_mem_pos_[size] + len;
    } else {
      source_mem_pos_[size + 1] = source_mem_pos_[size];
    }
  }
  return vid_mgr_->Add(meta_info_->size_++, docid);
}

int RawVector::Update(int docid, struct Field &field) {
  if (vid_mgr_->MultiVids() || docid >= (int)meta_info_->Size()) {
    return -1;
  }

  int vid = docid;

  if (field.value.size() / data_size_ <= 0) {
    LOG(ERROR) << "Doc [" << docid << "] len " << field.value.size() << "]";
    return -1;
  }

  if (UpdateToStore(vid, (uint8_t *)field.value.c_str(), field.value.size())) {
    LOG(ERROR) << "update to store error, docid=" << docid;
    return -1;
  }

  updated_vids_->enqueue(vid);
  // TODO: update source
  return 0;
}

int RawVector::Compress(uint8_t *v, ScopeVector &svec) {
#ifdef WITH_ZFP
  if (store_params_->compress_) {
    uint8_t *cmprs_v = nullptr;
    if (zfp_compressor_.Compress((float *)v, cmprs_v)) {
      return INTERNAL_ERR;
    }
    svec.Set(cmprs_v, true);
  } else
#endif
  {
    svec.Set(v, false);
  }
  return 0;
}

int RawVector::Decompress(uint8_t *cmprs_v, int n, uint8_t *&vec,
                          bool &deletable) const {
#ifdef WITH_ZFP
  if (store_params_->compress_) {
    float *v = nullptr;
    if (zfp_compressor_.Decompress(cmprs_v, n, v)) {
      return INTERNAL_ERR;
    }
    vec = (uint8_t *)v;
    deletable = true;
  } else
#endif
  {
    vec = cmprs_v;
    deletable = false;
  }
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

void AsyncFlusher::Until(int nexpect) const {
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

  if (!jp.GetInt("segment_size", segment_size_)) {
    if (segment_size_ <= 0) {
      LOG(ERROR) << "invalid segment size=" << segment_size_;
      return -1;
    }
  }

  if (jp.Contains("compress") && jp.GetBool("compress", compress_)) {
    LOG(ERROR) << "invalid compress value";
    return -1;
  }

  return 0;
}

void StartFlushingIfNeed(RawVector *vec) {
  AsyncFlusher *flusher = nullptr;
  if ((flusher = dynamic_cast<AsyncFlusher *>(vec))) {
    flusher->Start();
    const std::string &name = vec->MetaInfo()->Name();
    LOG(INFO) << "start flushing, raw vector=" << name;
  }
}

void StopFlushingIfNeed(RawVector *vec) {
  AsyncFlusher *flusher = dynamic_cast<AsyncFlusher *>(vec);
  if (flusher) {
    flusher->Until(vec->GetVectorNum());
    flusher->Stop();
    const std::string &name = vec->MetaInfo()->Name();
    LOG(INFO) << "stop flushing, raw vector=" << name;
  }
}

}  // namespace tig_gamma
