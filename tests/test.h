/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/utils.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "api_data/gamma_config.h"
#include "bitmap.h"
#include "c_api/api_data/gamma_response.h"
#include "c_api/gamma_api.h"
#include "log.h"
#include "util/utils.h"

using std::string;

namespace {

string kIVFPQParam =
    "{\"nprobe\" : 10, \"metric_type\" : \"InnerProduct\", \"ncentroids\" : "
    "256,\"nsubvector\" : 64, \"relayout_group_size\": 4}";

string kHNSWParam_str =
    "{\"nlinks\" : 32, \"metric_type\" : \"InnerProduct\", \"efSearch\" : "
    "64,\"efConstruction\" : 160}";

string kFLATParam_str = "{\"metric_type\" : \"InnerProduct\"}";

struct Options {
  Options() {
    nprobe = 10;
    doc_id = 0;
    d = 512;
    filter = true;
    print_doc = true;
    search_thread_num = 1;
    max_doc_size = 10000 * 200;
    add_doc_num = 10000 * 20;
    search_num = 100;
    indexing_size = 10000 * 1;
    fields_vec = {"_id", "img", "field1", "field2", "field3"};
    fields_type = {tig_gamma::DataType::STRING, tig_gamma::DataType::STRING,
                   tig_gamma::DataType::STRING, tig_gamma::DataType::INT,
                   tig_gamma::DataType::INT};
    vector_name = "abc";
    path = "files";
    log_dir = "log";
    model_id = "model";
    retrieval_type = "IVFPQ";
    store_type = "MemoryOnly";
    // store_type = "RocksDB";
    profiles.resize(max_doc_size * fields_vec.size());
    engine = nullptr;
    add_type = 0;
  }

  int nprobe;
  size_t doc_id;
  size_t doc_id2;
  size_t d;
  size_t max_doc_size;
  size_t add_doc_num;
  size_t search_num;
  int indexing_size;
  bool filter;
  bool print_doc;
  int search_thread_num;
  std::vector<string> fields_vec;
  std::vector<tig_gamma::DataType> fields_type;
  string path;
  string log_dir;
  string vector_name;
  string model_id;
  string retrieval_type;
  string store_type;
  int add_type;  // 0 single add, 1 batch add

  std::vector<string> profiles;
  float *feature;

  string profile_file;
  string feature_file;
  char *docids_bitmap_;
  void *engine;
};

void printDoc(struct tig_gamma::ResultItem &result_item, std::string &msg,
              struct Options &opt) {
  msg += string("score [") + std::to_string(result_item.score) + "], ";
  for (size_t i = 0; i < result_item.names.size(); ++i) {
    std::string &name = result_item.names[i];
    tig_gamma::DataType data_type;
    for (size_t j = 0; j < opt.fields_vec.size(); ++j) {
      if (name == opt.vector_name) {
        data_type = tig_gamma::DataType::VECTOR;
        break;
      }
      if (name == opt.fields_vec[j]) {
        data_type = opt.fields_type[j];
        break;
      }
    }
    if (name == "float") {
      data_type = tig_gamma::DataType::FLOAT;
    }

    msg += "field name [" + name + "], type [" +
           std::to_string(static_cast<int>(data_type)) + "], value [";
    std::string &value = result_item.values[i];
    if (data_type == tig_gamma::DataType::INT) {
      msg += std::to_string(*((int *)value.data()));
    } else if (data_type == tig_gamma::DataType::LONG) {
      msg += std::to_string(*((long *)value.data()));
    } else if (data_type == tig_gamma::DataType::FLOAT) {
      msg += std::to_string(*((float *)value.data()));
    } else if (data_type == tig_gamma::DataType::DOUBLE) {
      msg += std::to_string(*((double *)value.data()));
    } else if (data_type == tig_gamma::DataType::STRING) {
      msg += value;
    } else if (data_type == tig_gamma::DataType::VECTOR) {
      std::string str_vec;
      int d = -1;
      memcpy((void *)&d, value.data(), sizeof(int));

      d /= sizeof(float);
      int cur = sizeof(int);

      const float *feature =
          reinterpret_cast<const float *>(value.data() + cur);

      cur += d * sizeof(float);
      int len = value.length();
      char source[len - cur];

      memcpy(source, value.data() + cur, len - cur);

      for (int i = 0; i < d; ++i) {
        str_vec += std::to_string(feature[i]) + ",";
      }
      str_vec.pop_back();

      std::string source_str = std::string(source, len - cur);
      msg += str_vec + "], source [" + source_str + "]";
    }
    msg += "], ";
  }
}

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
  FILE *f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d;
  fread(&d, 1, sizeof(int), f);
  assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  fseek(f, 0, SEEK_SET);
  struct stat st;
  fstat(fileno(f), &st);
  size_t sz = st.st_size;
  assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  size_t n = sz / ((d + 1) * 4);

  *d_out = d;
  *n_out = n;
  float *x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");

  // shift array to remove row headers
  for (size_t i = 0; i < n; i++)
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

  fclose(f);
  return x;
}

int CreateFile(std::string &path, bool truncate = true) {
  int flags = O_WRONLY | O_CREAT;
  if (truncate) {
    flags |= O_TRUNC;
  }
  int fd = open(path.c_str(), flags, 00777);
  assert(fd != -1);
  close(fd);
  return 0;
}

struct FileHelper {
  std::string file_path;
  FILE *fp;

  FileHelper(std::string path) { file_path = path; }

  ~FileHelper() {
    if (fp) fclose(fp);
  }

  int Open(const char *mode) {
    fp = fopen(file_path.c_str(), mode);
    if (fp == nullptr) return -1;
    return 0;
  }

  size_t Read(void *data, size_t len) { return fread(data, 1, len, fp); }
};

string GetCurrentCaseName() {
  const ::testing::TestInfo *const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  return string(test_info->test_case_name()) + "_" + test_info->name();
}

void Sleep(long milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

struct RandomGenerator {
  RandomGenerator() {
    std::srand(std::time(nullptr));
    srand48(std::time(nullptr));
  }
  int Rand(int n, int offset = 0) { return std::rand() % n + offset; }
  double RandDouble(double offset = 0.0f) { return drand48() + offset; }
};

float random_float(float min, float max, unsigned int seed = 0) {
  static std::default_random_engine e(seed);
  static std::uniform_real_distribution<float> u(min, max);
  return u(e);
}

}  // namespace
