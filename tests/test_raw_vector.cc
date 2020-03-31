/**
 * Copyright (c) The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "mmap_raw_vector.h"

#ifdef WITH_ROCKSDB
#include "rocksdb_raw_vector.h"
#endif  // WITH_ROCKSDB

#include <fcntl.h>
#include <gtest/gtest.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include "raw_vector_factory.h"
#include "test.h"
#include "utils.h"
#include "vector_file_mapper.h"

using namespace tig_gamma;
using namespace std;

float *BuildVector(int dim, float offset) {
  float *v = new float[dim];
  for (int i = 0; i < dim; i++) {
    v[i] = i + offset;
  }
  return v;
}

float *BuildVectors(int n, int dim, float offset) {
  float *vecs = new float[dim * n];
  for (int i = 0; i < n; i++) {
    float *v = vecs + dim * i;
    for (int j = 0; j < dim; j++) {
      v[j] = j + i + offset;
    }
  }
  return vecs;
}

Field *BuildVectorField(int dim, float offset) {
  float *data = BuildVector(dim, offset);
  Field *field =
      MakeField(nullptr, MakeByteArray((char *)data, sizeof(float) * dim),
                nullptr, VECTOR);
  delete[] data;
  return field;
}

bool floatArrayEquals(const float *a, int m, const float *b, int n) {
  if (m != n) return false;
  for (int i = 0; i < m; i++) {
    if (std::fabs(a[i] - b[i]) > 0.0001f) {
      return false;
    }
  }
  return true;
}

void AddToRawVector(RawVector *raw_vector, int start_id, int num,
                    int dimension) {
  int end = start_id + num;
  for (int i = start_id; i < end; i++) {
    Field *field = BuildVectorField(dimension, i);
    int ret = raw_vector->Add(i, field);
    assert(0 == ret);
    DestroyField(field);
  }
}

void ValidateVector(RawVector *raw_vector, int start_id, int num,
                    int dimension) {
  for (int i = start_id; i < num; i++) {
    const float *expect = BuildVector(dimension, i);
    ScopeVector scope_vec;
    raw_vector->GetVector(i, scope_vec);
    const float *peek_vector = scope_vec.Get();
    ASSERT_TRUE(floatArrayEquals(expect, dimension, peek_vector, dimension))
        << "******GetVector float array equal error, vid=" << i << ", peek=["
        << peek_vector[0] << ", " << peek_vector[1] << "]"
        << ", expect=[" << expect[0] << ", " << expect[1] << "]";
    delete[] expect;
  }
}

void ValidateVectorHeader(RawVector *raw_vector, int start_id, int num,
                          int dimension) {
  ScopeVector scope_vec;
  raw_vector->GetVectorHeader(start_id, start_id + num, scope_vec);
  const float *vectors = scope_vec.Get();
  for (int i = start_id; i < start_id + num; i++) {
    const float *expect = BuildVector(dimension, i);
    const float *peek_vector = vectors + (i - start_id) * dimension;
    ASSERT_TRUE(floatArrayEquals(expect, dimension, peek_vector, dimension))
        << "******GetVector float array equal error, vid=" << i << ", peek=["
        << peek_vector[0] << ", " << peek_vector[1] << "]"
        << ", expect=[" << expect[0] << ", " << expect[1] << "]";
    delete[] expect;
  }
}

TEST(FileMapper, Normal) {
  string file_path = "test_file_maper.fet";
  int offset = 0;
  int max_size = 1000;
  int dimension = 512;

  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 00777);
  assert(-1 != fd);
  close(fd);

  VectorFileMapper *mapper =
      new VectorFileMapper(file_path, offset, max_size, dimension);
  assert(0 == mapper->Init());
  assert(0 == mapper->GetMappedNum());

  int feature_size = sizeof(float) * dimension;
  fd = open(file_path.c_str(), O_WRONLY | O_APPEND);
  for (int i = 0; i < max_size; i++) {
    float *feature = BuildVector(dimension, i);
    assert(feature_size == write(fd, (void *)feature, feature_size));
    delete[] feature;
    cerr << "i=" << i << endl;
  }

  const float *features = mapper->GetVectors();
  assert(nullptr != features);
  for (int i = 0; i < max_size; i++) {
    const float *map_feature = features + (dimension * i);
    float *expect = BuildVector(dimension, i);
    cerr << "float array equal, i=" << i << ", feature=[" << map_feature[0]
         << ", " << map_feature[1] << ", " << map_feature[2] << "]"
         << ", expect=[" << expect[0] << ", " << expect[1] << ", " << expect[2]
         << "]" << endl;
    assert(floatArrayEquals(expect, dimension, map_feature, dimension));
    delete[] expect;
  }

  delete mapper;
}

void TestFileMapperLoad() {
  string file_path = "test.fet";
  int offset = 9;
  int max_size = 1000000;
  int dimension = 512;
  VectorFileMapper *mapper =
      new VectorFileMapper(file_path, offset, max_size, dimension);
  assert(0 == mapper->Init());
  int map_num = mapper->GetMappedNum();
  const float *features = mapper->GetVectors();
  cerr << "map_num=" << map_num << endl;
  const float *feature = new float[dimension];
  int fea_len = sizeof(float) * dimension;
  double begin = utils::getmillisecs();
  for (int i = 0; i < map_num; i++) {
    memcpy((void *)feature, features + i * dimension, fea_len);
    cerr << "id=" << i << ", feature[0]=" << feature[0] << endl;
  }
  cerr << "memory copy finished, cost=" << utils::getmillisecs() - begin
       << endl;
}

void TestFileMapperRandRead() {
  string file_path = "test.fet";
  int offset = 9;
  int max_size = 1000000;
  int dimension = 512;
  VectorFileMapper *mapper =
      new VectorFileMapper(file_path, offset, max_size, dimension);
  assert(0 == mapper->Init());
  std::this_thread::sleep_for(std::chrono::milliseconds(20000));
  int map_num = mapper->GetMappedNum();
  cerr << "mmap finished, map num=" << map_num << endl;
  const float *features = mapper->GetVectors();
  int times = 1000000;
  std::srand(std::time(nullptr));
  const float *feature = new float[dimension];
  int fea_len = sizeof(float) * dimension;
  double begin = utils::getmillisecs();
  for (int i = 0; i < times; i++) {
    int id = std::rand();
    id = id % map_num;
    memcpy((void *)feature, features + id * dimension, fea_len);
    cerr << "i=" << i << ", id=" << id << ", feature[0]=" << feature[0] << endl;
  }
  cerr << "file mapper random read finished, cost="
       << utils::getmillisecs() - begin << "ms, times=" << times << endl;
}

TEST(MmapRawVector, Normal) {
  string root_path = "./" + GetCurrentCaseName();
  string name = "abc";
  string file_path = root_path + "/" + name + ".fet";
  int max_size = 100000;
  int dimension = 511;

  utils::remove_dir(root_path.c_str());
  utils::make_dir(root_path.c_str());

  CreateFile(file_path);

  RawVector *raw_vector = RawVectorFactory::Create(
      Mmap, name, dimension, max_size, root_path, "{\"cache_size\": 4}");
  assert(0 == raw_vector->Init());
  StartFlushingIfNeed(raw_vector);
  int doc_num = 50000;
  int ret = -1;
  for (int i = 0; i < doc_num; i++) {
    if (i % 10000 == 0) cerr << "add i=" << i << endl;
    Field *field = BuildVectorField(dimension, i);
    ret = raw_vector->Add(i, field);
    assert(0 == ret);
    DestroyField(field);
  }

  int added_doc_num = raw_vector->GetVectorNum();
  assert(doc_num == added_doc_num);
  ValidateVectorHeader(raw_vector, 0, added_doc_num, dimension);
  ASSERT_EQ(0, raw_vector->Dump("", 0, doc_num - 1));
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;

  raw_vector =
      RawVectorFactory::Create(Mmap, name, dimension, max_size, root_path, "");
  assert(nullptr != raw_vector);
  assert(0 == raw_vector->Init());
  vector<string> paths;
  ASSERT_EQ(0, raw_vector->Load(paths, doc_num));
  ValidateVector(raw_vector, 0, doc_num, dimension);
}

TEST(MmapRawVector, DumpLoad) {
  string root_path = GetCurrentCaseName();
  string name = "abc";
  string file_path = root_path + "/" + name + ".fet";
  int max_size = 10000;
  int dimension = 512;

  utils::remove_dir(root_path.c_str());
  utils::make_dir(root_path.c_str());
  CreateFile(file_path);

  RawVector *raw_vector =
      RawVectorFactory::Create(Mmap, name, dimension, max_size, root_path, "");
  ASSERT_EQ(0, raw_vector->Init());
  StartFlushingIfNeed(raw_vector);

  int doc_num = 500;
  AddToRawVector(raw_vector, 0, doc_num, dimension);

  ASSERT_EQ(doc_num, raw_vector->GetVectorNum());
  ValidateVectorHeader(raw_vector, 0, doc_num, dimension);

  ASSERT_EQ(0, raw_vector->Dump(root_path + "/dump/1", 0, doc_num - 1));
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;

  vector<string> paths;

  // load: load_num == disk_doc_num;
  LOG(INFO) << "---------------load all----------------";
  int load_num = doc_num;
  raw_vector =
      RawVectorFactory::Create(Mmap, name, dimension, max_size, root_path, "");
  ASSERT_NE(nullptr, raw_vector);
  ASSERT_EQ(0, raw_vector->Init());
  StartFlushingIfNeed(raw_vector);
  ASSERT_EQ(0, raw_vector->Load(paths, load_num));
  ValidateVector(raw_vector, 0, load_num, dimension);
  ASSERT_EQ(load_num, raw_vector->GetVectorNum());
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;

  LOG(INFO) << "---------------load some----------------";
  // load: load_num < disk_doc_num;
  load_num = doc_num - 100;
  raw_vector =
    RawVectorFactory::Create(Mmap, name, dimension, max_size, root_path, "");
  ASSERT_NE(nullptr, raw_vector);
  ASSERT_EQ(0, raw_vector->Init());
  StartFlushingIfNeed(raw_vector);
  ASSERT_EQ(0, raw_vector->Load(paths, load_num));
  ValidateVector(raw_vector, 0, load_num, dimension);
  ASSERT_EQ(load_num, raw_vector->GetVectorNum());
  ASSERT_EQ(load_num * sizeof(int), utils::get_file_size(root_path + "/" + name + ".docid"));
  ASSERT_EQ(load_num * dimension * sizeof(float), utils::get_file_size(root_path + "/" + name + ".fet"));
  ASSERT_EQ((load_num + 1) * sizeof(long), utils::get_file_size(root_path + "/" + name + ".src.pos"));
  char *source1 = nullptr, *source2 = nullptr;
  int len1 = 0, len2 = 0;
  raw_vector->GetSource(0, source1, len1);
  raw_vector->GetSource(load_num - 1, source2, len2);
  ASSERT_EQ(source2 + len2 - source1, utils::get_file_size(root_path + "/" + name + ".src"));

  LOG(INFO) << "---------------dump after load and add----------------";
  // dump after load and add
  int add_num = 200;
  AddToRawVector(raw_vector, load_num, add_num, dimension);
  ASSERT_EQ(0, raw_vector->Dump(root_path + "/dump/2", load_num, load_num + add_num - 1));
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;

  LOG(INFO) << "---------------reload after dump----------------";
  load_num = load_num + add_num;
  raw_vector =
    RawVectorFactory::Create(Mmap, name, dimension, max_size, root_path, "");
  ASSERT_NE(nullptr, raw_vector);
  ASSERT_EQ(0, raw_vector->Init());
  StartFlushingIfNeed(raw_vector);
  ASSERT_EQ(0, raw_vector->Load(paths, load_num));
  ValidateVector(raw_vector, 0, load_num, dimension);
  ASSERT_EQ(load_num, raw_vector->GetVectorNum());
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;

}

int CreateFeatureFile(string file_path, int max_size, int dimension) {
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 00777);
  assert(-1 != fd);
  close(fd);

  int feature_size = sizeof(float) * dimension;
  fd = open(file_path.c_str(), O_WRONLY | O_APPEND);
  for (int i = 0; i < max_size; i++) {
    float *feature = BuildVector(dimension, i + 0.1f);
    assert(feature_size == write(fd, (void *)feature, feature_size));
    delete[] feature;
    if (i % 10000 == 0) {
      cerr << "create feature file i=" << i << endl;
    }
  }
  close(fd);
  cerr << "create feature file success, file path=" << file_path
       << ", max size=" << max_size << ", dimension=" << dimension << endl;
  return 0;
}

void TestMemoryDiskRawFeatureRandomGet(int m_size, int r_times) {
  string file_path = "test_memory_disk_random_get.fet";
  int max_size = 10000;
  int dimension = 512;
  int read_times = 10000;
  int fea_len = sizeof(float) * dimension;
  if (m_size != 0) {
    max_size = m_size;
  }
  if (r_times != 0) {
    read_times = r_times;
  }
  if (access(file_path.c_str(), F_OK) != 0) {
    cerr << "file_path=" << file_path << " is not existed, create it" << endl;
    assert(0 == CreateFeatureFile(file_path, max_size, dimension));
  } else {
    cerr << "file_path=" << file_path << " is already existed, reuse it!"
         << endl;
    long file_size = utils::get_file_size(file_path.c_str());
    if (file_size % fea_len != 0) {
      cerr << "invalid file size=" << file_size << endl;
      assert(0 == 1);
    }
    max_size = file_size / fea_len;
  }
  int max_buffer_size = m_size * 0.7;
  cerr << "file_path=" << file_path << ", max_size=" << max_size
       << ", dimension=" << dimension << ", read times=" << read_times
       << ", max buffer size=" << max_buffer_size << endl;
  StoreParams params;
  params.cache_size_ =
      max_buffer_size * dimension * sizeof(float) / (1024 * 1024);

  RawVector *raw_feature = new MmapRawVector("test_memory_disk_random_get",
                                             dimension, max_size, "./", params);
  // raw_feature->SetFlushBatchSize(500);
  assert(0 == raw_feature->Init());
  StartFlushingIfNeed(raw_feature);

  int *ids = new int[max_size];
  std::srand(std::time(nullptr));
  // float *feature = new float[dimension];
  for (int i = 0; i < read_times; i++) {
    int id = std::rand();
    id = id % max_size;
    ids[i] = id;
  }

  double begin = utils::getmillisecs();
  for (int i = 0; i < read_times; i++) {
    int id = ids[i];
    ScopeVector scope_vec;
    raw_feature->GetVector(id, scope_vec);
    const float *feature = scope_vec.Get();
    if (i % 10000 == 0) {
      cerr << "i=" << i << ", id=" << id << ", feature[0]=" << feature[0]
           << endl;
    }
  }
  delete[] ids;
  StopFlushingIfNeed(raw_feature);
  delete raw_feature;
  cerr << "rand read finished, times=" << read_times
       << ", cost=" << utils::getmillisecs() - begin << "ms" << endl;
}

TEST(VectorBufferQueue, Normal) {
  int dimension = 512;
  int max_buffer_size = 2000;
  int chunk_num = 10;

  VectorBufferQueue *queue =
      new VectorBufferQueue(max_buffer_size, dimension, chunk_num);
  assert(0 == queue->Init());
  assert(0 == queue->GetPopSize());
  assert(0 == queue->Size());

  int doc_num = 2000;
  for (int i = 0; i < doc_num; i++) {
    float *feature = BuildVector(dimension, i);
    // cerr << "add i=" << i << endl;
    assert(0 == queue->Push(feature, dimension, -1));
    delete[] feature;
  }
  assert(doc_num == queue->GetPopSize());
  assert(doc_num == queue->Size());

  int peek_num = 1000;
  float *peek_features = new float[peek_num * dimension];
  assert(0 == queue->Pop(peek_features, dimension, peek_num, -1));
  for (int i = 0; i < peek_num; i++) {
    float *expect = BuildVector(dimension, i);
    const float *peek_feature = peek_features + i * dimension;
    cerr << "float array equal, i=" << i << ", feature=[" << peek_feature[0]
         << ", " << peek_feature[1] << ", " << peek_feature[2] << "]"
         << ", expect=[" << expect[0] << ", " << expect[1] << ", " << expect[2]
         << "]" << endl;
    assert(floatArrayEquals(expect, dimension, peek_feature, dimension));
    delete[] expect;
  }
  assert(doc_num - peek_num == queue->GetPopSize());
  assert(doc_num == queue->Size());
  delete[] peek_features;
  peek_features = nullptr;

  int readd_num = 500;
  for (int i = 0; i < readd_num; i++) {
    float *feature = BuildVector(dimension, i + doc_num);
    // cerr << "add i=" << i << endl;
    assert(0 == queue->Push(feature, dimension, -1));
    delete[] feature;
  }

  assert(doc_num - peek_num + readd_num == queue->GetPopSize());
  assert(doc_num == queue->Size());

  int peek_num_2 = doc_num - peek_num + readd_num;
  peek_features = new float[peek_num_2 * dimension];
  assert(0 == queue->Pop(peek_features, dimension, peek_num_2, -1));
  for (int i = 0; i < peek_num_2; i++) {
    float *expect = BuildVector(dimension, i + peek_num);
    const float *peek_feature = peek_features + i * dimension;
    cerr << "float array equal, i=" << i << ", feature=[" << peek_feature[0]
         << ", " << peek_feature[1] << ", " << peek_feature[2] << "]"
         << ", expect=[" << expect[0] << ", " << expect[1] << ", " << expect[2]
         << "]" << endl;
    assert(floatArrayEquals(expect, dimension, peek_feature, dimension));
    delete[] expect;
  }
  assert(0 == queue->GetPopSize());
  assert(doc_num == queue->Size());
  delete queue;
}

TEST(VectorBufferQueue, RandRead) {
  int dimension = 512;
  int max_buffer_size = 20 * 10000;
  int read_times = 10 * 10000;
  int chunk_num = 1000;

  VectorBufferQueue *queue =
      new VectorBufferQueue(max_buffer_size, dimension, chunk_num);
  assert(0 == queue->Init());
  assert(0 == queue->GetPopSize());
  assert(0 == queue->Size());

  int doc_num = max_buffer_size;
  for (int i = 0; i < doc_num; i++) {
    float *feature = BuildVector(dimension, i);
    // cerr << "add i=" << i << endl;
    assert(0 == queue->Push(feature, dimension, -1));
    delete[] feature;
  }
  assert(doc_num == queue->GetPopSize());
  assert(doc_num == queue->Size());

  std::srand(std::time(nullptr));
  float *feature = new float[dimension];
  double begin = utils::getmillisecs();
  for (int i = 0; i < read_times; i++) {
    int id = std::rand();
    id = id % doc_num;
    assert(0 == queue->GetVector(id, feature, dimension));
    if (i % 10000 == 0)
      cerr << "i=" << i << ", id=" << id << ", feature[0]=" << feature[0]
           << endl;
  }
  cerr << "rand read finished, times=" << read_times
       << ", cost=" << utils::getmillisecs() - begin << "ms" << endl;
}

TEST(VectorBufferQueue, BatchPush) {
  int dimension = 512;
  int max_buffer_size = 2000;
  int chunk_num = 10;

  VectorBufferQueue *queue =
      new VectorBufferQueue(max_buffer_size, dimension, chunk_num);
  assert(0 == queue->Init());

  int batch = max_buffer_size / chunk_num - 1;
  for (int i = 0; i < chunk_num; i++) {
    float *feature = BuildVectors(batch, dimension, i * batch);
    // cerr << "add i=" << i << endl;
    assert(0 == queue->Push(feature, dimension, batch, -1));
    delete[] feature;
  }

  int peek_num = batch * chunk_num;
  cerr << "peek num=" << peek_num << endl;
  float *peek_features = new float[peek_num * dimension];
  assert(0 == queue->Pop(peek_features, dimension, peek_num, -1));
  float *expects = BuildVectors(peek_num, dimension, 0);
  for (int i = 0; i < peek_num; i++) {
    float *expect = expects + i * dimension;
    const float *peek_feature = peek_features + i * dimension;
    ASSERT_TRUE(floatArrayEquals(expect, dimension, peek_feature, dimension))
        << "i=" << i << ", feature=[" << peek_feature[0] << ", "
        << peek_feature[1] << "], expect=[" << expect[0] << ", " << expect[1]
        << "]";
  }
  delete[] peek_features;
  delete[] expects;
  delete queue;
}

int added_num = 0;
void AddFunc(VectorBufferQueue *qu, int check_num, int dim) {
  cerr << "****AddFunc: check num=" << check_num << ", dimension=" << dim
       << endl;
  srand(time(NULL));
  for (int i = 0; i < check_num; i++) {
    float *v = BuildVector(dim, i);
    assert(0 == qu->Push(v, dim, -1));
    added_num++;
    int wait = rand() % 10;
    cout << "****AddFunc: add vector, i=" << i << ", size=" << qu->Size()
         << "peek size=" << qu->GetPopSize() << ", next wait=" << wait << "ms"
         << endl;
    delete[] v;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }
}

int flushed_num = 0;
int Flush(VectorBufferQueue *feature_buffer_queue, int check_num, int dimension,
          int flush_batch) {
  cerr << "****FlushFunc: check num=" << check_num
       << ", dimension=" << dimension << ", flush batch=" << flush_batch;
  int checked = 0;
  float *flush_batch_features = new float[flush_batch * dimension];
  while (checked < check_num) {
    try {
      int poll_size = feature_buffer_queue->GetPopSize();
      int poll_num = poll_size > flush_batch ? flush_batch : poll_size;
      if (poll_num > 0) {
        cerr << "peek number=" << poll_num << endl;
        // error handle
        assert(0 == feature_buffer_queue->Pop(flush_batch_features, dimension,
                                              poll_num, -1));
        flushed_num += poll_num;
        for (int i = 0; i < poll_num; i++) {
          float *expect = BuildVector(dimension, checked);
          const float *peek_feature = flush_batch_features + i * dimension;
          cerr << "Flush float array equal, checked=" << checked
               << ", feature=[" << peek_feature[0] << ", " << peek_feature[1]
               << ", " << peek_feature[2] << "]"
               << ", expect=[" << expect[0] << ", " << expect[1] << ", "
               << expect[2] << "]" << endl;
          assert(floatArrayEquals(expect, dimension, peek_feature, dimension));
          checked++;
          delete[] expect;
        }

        cerr << "flush one batch features to disk success! peek size="
             << poll_size << ", peek number=" << poll_num
             << ", max flushed feature id=" << checked << endl;

      } else {
        cerr << "no feature need to flush, buffer queue size="
             << feature_buffer_queue->Size() << endl;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } catch (const std::exception &e) {
      cerr << "Flush exception: " << e.what() << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  return 0;
}

void GetFunc(VectorBufferQueue *queue, int num, int dimension,
             int buffer_size) {
  srand(time(NULL));
  float peek_feature[dimension];
  cerr << "******GetFunc num=" << num << ", dimension=" << dimension << endl;
  for (int i = 0; i < num; i++) {
    int a = added_num;
    int base = a > buffer_size ? a - buffer_size : 0;
    int vid = std::rand() % buffer_size + base;
    int ret = queue->GetVector(vid, peek_feature, dimension);
    if (ret == 0) {
      float *expect = BuildVector(dimension, vid);
      cerr << "******GetFunc float array equal, i=" << i << ", vid=" << vid
           << ", feature=[" << peek_feature[0] << ", " << peek_feature[1]
           << ", " << peek_feature[2] << "]"
           << ", expect=[" << expect[0] << ", " << expect[1] << ", "
           << expect[2] << "]" << endl;
      assert(floatArrayEquals(expect, dimension, peek_feature, dimension));
      delete[] expect;
    } else {
      cerr << "******GetFunc failed, vid=" << vid << endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void TestVectorBufferQueueTwoThreads() {
  int dimension = 512;
  int max_buffer_size = 200;
  int check_num = 5000;
  int flush_batch = 20;
  int chunk_num = 20;
  // string file_path = "abc.fet";

  // CreateFile(file_path);

  VectorBufferQueue *queue =
      new VectorBufferQueue(max_buffer_size, dimension, chunk_num);
  assert(0 == queue->Init());
  std::thread addThread(AddFunc, queue, check_num, dimension);
  std::thread flushThread(Flush, queue, check_num, dimension, flush_batch);
  std::thread getThread(GetFunc, queue, check_num, dimension, max_buffer_size);
  addThread.join();
  flushThread.join();
  getThread.join();
  delete queue;
}

#ifdef WITH_ROCKSDB

TEST(RocksDBRawVector, Normal) {
  string name = "test_rocks";
  string root_path = "./test_rocks";
  int dim = 512;
  StoreParams params;
  params.cache_size_ = 256;

  utils::remove_dir(root_path.c_str());
  utils::make_dir(root_path.c_str());

  RawVector *raw_vector =
      new RocksDBRawVector(name, dim, 20000, root_path, params);
  int ret = raw_vector->Init();
  assert(0 == ret);
  StartFlushingIfNeed(raw_vector);
  int doc_num = 1234;
  AddToRawVector(raw_vector, 0, doc_num, dim);
  ValidateVector(raw_vector, 0, doc_num, dim);
  ValidateVectorHeader(raw_vector, 0, doc_num, dim);
  string dump_path = ""; // do not need it
  int dump_docid = 0, max_dump_docid = doc_num - 1;
  ASSERT_EQ(0, raw_vector->Dump(dump_path, dump_docid, max_dump_docid));
  StopFlushingIfNeed(raw_vector);
  dump_docid = max_dump_docid + 1;
  delete raw_vector;

  raw_vector = new RocksDBRawVector(name, dim, 20000, root_path, params);
  ret = raw_vector->Init();
  ASSERT_EQ(0, ret);
  StartFlushingIfNeed(raw_vector);
  vector<string> load_paths;
  ASSERT_EQ(0, raw_vector->Load(load_paths, doc_num));
  AddToRawVector(raw_vector, doc_num, 100, dim);
  ValidateVectorHeader(raw_vector, 0, doc_num + 100, dim);
  ValidateVector(raw_vector, 0, doc_num + 100, dim);
  StopFlushingIfNeed(raw_vector);
  max_dump_docid = doc_num + 100 - 1;
  ASSERT_EQ(0, raw_vector->Dump(dump_path, dump_docid, max_dump_docid));
  dump_docid = max_dump_docid + 1;
  delete raw_vector;

  raw_vector = new RocksDBRawVector(name, dim, 20000, root_path, params);
  ret = raw_vector->Init();
  ASSERT_EQ(0, ret);
  StartFlushingIfNeed(raw_vector);
  ASSERT_EQ(0, raw_vector->Load(load_paths, doc_num + 100));
  AddToRawVector(raw_vector, doc_num + 100, 100, dim);
  ValidateVectorHeader(raw_vector, 0, doc_num + 200, dim);
  ValidateVector(raw_vector, 0, doc_num + 200, dim);
  StopFlushingIfNeed(raw_vector);
  max_dump_docid = doc_num + 200 - 1;
  ASSERT_EQ(0, raw_vector->Dump(dump_path, dump_docid, max_dump_docid));
  dump_docid = max_dump_docid + 1;
  delete raw_vector;
}

void TestRocksDBRawVectorRandomGet(int max_size, int read_times) {
  string name = "test_rocks_random_get";
  string root_path = "./test_rocks_random_get";
  int dim = 512;
  StoreParams params;
  params.cache_size_ = 1024;
  int existed = access(root_path.c_str(), F_OK);
  cerr << "max_size=" << max_size << ", read_times=" << read_times
       << ", path existed=" << existed;
  RawVector *raw_vector =
      new RocksDBRawVector(name, dim, max_size, root_path, params);
  int ret = raw_vector->Init();
  assert(0 == ret);
  StartFlushingIfNeed(raw_vector);
  if (existed != 0) {
    for (int i = 0; i < max_size; i++) {
      Field *field = BuildVectorField(dim, i);
      ret = raw_vector->Add(i, field);
      assert(0 == ret);
      DestroyField(field);
      if (i % 10000 == 0) {
        cerr << "add vector i=" << i << endl;
      }
    }
  }
  int *ids = new int[read_times];
  std::srand(std::time(nullptr));
  // float *feature = new float[dimension];
  for (int i = 0; i < read_times; i++) {
    int id = std::rand();
    id = id % max_size;
    ids[i] = id;
  }

  double begin = utils::getmillisecs();
  for (int i = 0; i < read_times; i++) {
    int id = ids[i];
    ScopeVector scope_vec;
    raw_vector->GetVector(id, scope_vec);
    const float *feature = scope_vec.Get();
    if (i % 10000 == 0) {
      cerr << "random get i=" << i << ", id=" << id
           << ", feature[0]=" << feature[0] << endl;
    }
  }
  delete[] ids;
  cerr << "rand read finished, times=" << read_times
       << ", cost=" << utils::getmillisecs() - begin << "ms" << endl;
  StopFlushingIfNeed(raw_vector);
  delete raw_vector;
}

#endif  // WITH_ROCKSDB

void PrintUsage() {
  cerr << "Usage: test_mixed_feature_house case_id" << endl;
  cerr << "Example: test_mixed_feature_house 1" << endl;
  cerr << "\t 1:TestFileMapper" << endl;
  cerr << "\t 2:TestFileMapperLoad" << endl;
  cerr << "\t 3:TestFileMapperRandRead" << endl;
  cerr << "\t 4:TestMmapRawVector" << endl;
  cerr << "\t 5:TestMmapRawVectorRandomGet max_size read_times" << endl;
#ifdef WITH_ROCKSDB
  cerr << "\t 6:TestRocksDBRawVector" << endl;
  cerr << "\t 7:TestRocksDBRawVectorRandomGet max_size read_times" << endl;
#endif  // WITH_ROCKSDB
  cerr << "\t 8:TestVectorBufferQueue" << endl;
  cerr << "\t 9:TestVectorBufferQueueTwoThreads" << endl;
  cerr << "\t 10:TestVectorBufferQueueRandRead" << endl;
}

int main(int argc, char *argv[]) {
  string log_dir = "./test_raw_vector_log";
  SetLogDictionary(tig_gamma::StringToByteArray(log_dir));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  // if (argc < 2) {
  //     PrintUsage();
  //     return -1;
  //   }
  //   int case_id = std::stoi(argv[1]);
  //   int max_size = 0;
  //   int read_times = 0;
  //   cerr << "case id=" << case_id << endl;
  //   switch (case_id) {
  //   case 1:
  //     TestFileMapper();
  //     break;
  //   case 2:
  //     TestFileMapperLoad();
  //     break;
  //   case 3:
  //     TestFileMapperRandRead();
  //     break;
  //   case 4:
  //     TestMmapRawVector();
  //     break;
  //   case 5:
  //     max_size = (int)std::strtol(argv[2], NULL, 10);
  //     read_times = (int)std::strtol(argv[3], NULL, 10);
  //     TestMemoryDiskRawFeatureRandomGet(max_size, read_times);
  //     break;
  // #ifdef WITH_ROCKSDB
  //   case 6:
  //     TestRocksDBRawVector();
  //     break;
  //   case 7:
  //     max_size = (int)std::strtol(argv[2], NULL, 10);
  //     read_times = (int)std::strtol(argv[3], NULL, 10);
  //     TestRocksDBRawVectorRandomGet(max_size, read_times);
  //     break;
  // #endif //  WITH_ROCKSDB
  //   case 8:
  //     TestVectorBufferQueue();
  //     break;
  //   case 9:
  //     TestVectorBufferQueueTwoThreads();
  //     break;
  //   case 10:
  //     TestVectorBufferQueueRandRead();
  //     break;
  //   default:
  //     PrintUsage();
  //     return -1;
  //   }
  //   return 0;
}
