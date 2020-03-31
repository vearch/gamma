/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cJSON.h"
#include "test.h"
#include "util/utils.h"

using std::string;
int nprobe = 50;

void PrintResponse(Response *response, int k = 10) {
  cerr << "response req_num=" << response->req_num << endl;
  for (int j = 0; j < response->req_num; j++) {
    struct SearchResult *search_result = response->results[j];
    printf("req id=%d, result number=%d, total=%d, code=%d\n", j,
           search_result->result_num, search_result->total,
           search_result->result_code);
    for (int i = 0; i < search_result->result_num; ++i) {
      struct ResultItem *result_item = GetResultItem(search_result, i);
      printf("req id=%d, i=%d, score [%f],  ", j, i, result_item->score);
      string msg = "";
      printDoc(result_item->doc, msg);
      string extra = "";
      if (result_item->extra) {
        extra = ByteArrayToString(result_item->extra);
      }
      printf("%s extra=%s \n\n", msg.c_str(), extra.c_str());
      if (i >= k - 1) break;
    }
  }
}

string GetDocField(Doc *doc) {
  for (int j = 0; j < doc->fields_num; ++j) {
    string field_name = ByteArrayToString(doc->fields[j]->name);
    if (field_name == "_id") {
      return ByteArrayToString(doc->fields[j]->value);
    }
  }
  return "";
}

void GetResponseDocIds(Response *response, int req_id,
                       std::vector<string> &doc_ids) {
  if (req_id >= response->req_num) {
    return;
  }
  struct SearchResult *search_result = response->results[req_id];
  for (int i = 0; i < search_result->result_num; ++i) {
    struct ResultItem *result_item = GetResultItem(search_result, i);
    string doc_id = GetDocField(result_item->doc);
    doc_ids.push_back(doc_id);
  }
}

template <typename T>
string VectorToString(T *v, int n) {
  std::stringstream msg;
  msg << "[";
  for (int i = 0; i < n; i++) {
    msg << v[i];
    if (i != n - 1) {
      msg << ",";
    }
  }
  msg << "]";
  return msg.str();
}

template <class K, class V>
string MapToString(std::map<K, V> m) {
  std::stringstream msg;
  msg << "{";
  for (auto ite = m.begin(); ite != m.end(); ite++) {
    msg << "[" << ite->first << ":" << ite->second << "]";
  }
  msg << "}";
  return msg.str();
}

void TestMultiUrl() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files_range_multi_url";
  string log_dir = "files_range_multi_url_log";
  int max_doc_size = 1000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  // _id = sku
  std::vector<string> fields_vec = {"_id", "cid1", "cid2"};
  std::vector<enum DataType> fields_type = {STRING, INT, INT};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  struct VectorInfo **vectors_info = MakeVectorInfos(1);
  string model_id = "model";
  string vector_name = "abc";
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  struct VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(vector_name), FLOAT, TRUE, d,
      StringToByteArray(model_id), StringToByteArray(retrieval_type),
      StringToByteArray(store_type), nullptr);
  SetVectorInfo(vectors_info, 0, vector_info);

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, 1, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();

  string profile_file = "/root/xiedabin/vector-db/build/sku_multi_url.txt";
  string feature_file = "/root/xiedabin/vector-db/build/multi_url_feat.dat";

  FILE *fp_feature = fopen(feature_file.c_str(), "rb");
  EXPECT_NE(fp_feature, nullptr);

  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> xb(d * 1);
  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 2);
    std::vector<string> profiles = utils::split(str, "\t");
    if (profiles.size() < 10) {
      cerr << "idx=" << idx << "invalid split size=" << profiles.size()
           << ", line=" << str << endl;
      break;
    }
    string url1 = profiles[8];
    string url2 = profiles[9];
    if (url1 == "" || url2 == "") {
      cerr << "idx=" << idx << "invalid url, url1=" << url1 << ", url2=" << url2
           << endl;
      break;
    }

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field =
          MakeField(name, value, StringToByteArray(string("aaaaa")), data_type);
      SetField(fields, i, field);
    }

    struct ByteArray *url1_name = StringToByteArray(vector_name);
    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *url1_value = FloatToByteArray(xb.data(), d);
    struct ByteArray *url1_source = StringToByteArray(url1);
    struct Field *url1_field =
        MakeField(url1_name, url1_value, url1_source, VECTOR);
    SetField(fields, fields_vec.size(), url1_field);

    struct ByteArray *url2_name = StringToByteArray(vector_name);
    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *url2_value = FloatToByteArray(xb.data(), d);
    struct ByteArray *url2_source = StringToByteArray(url2);
    struct Field *url2_field =
        MakeField(url2_name, url2_value, url2_source, VECTOR);
    SetField(fields, fields_vec.size() + 1, url2_field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 2);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    ++idx;
    if (idx > 100000) {
      break;
    }
  }
  fin.close();
  fclose(fp_feature);

  double add_time = utils::getmillisecs() - start;

  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  std::thread t(BuildIndex, engine);
  t.detach();

  cerr << "waiting to add.....";
  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  int vector_byte_size = d * sizeof(float);
  int total_dim = d * 2;
  float *multi_vector = new float[total_dim];
  memcpy((void *)multi_vector, (void *)xb.data(), vector_byte_size);
  memcpy((void *)(multi_vector + d), (void *)xb.data(), vector_byte_size);

  struct VectorQuery **vector_querys = MakeVectorQuerys(1);
  struct ByteArray *value = FloatToByteArray(multi_vector, total_dim);
  struct VectorQuery *vector_query = MakeVectorQuery(
      StringToByteArray(vector_name), value, 0, 1000000, 0.1, 0);
  SetVectorQuery(vector_querys, 0, vector_query);

  RangeFilter **range_filters = MakeRangeFilters(2);
  string cid1 = fields_vec[1];
  string cid2 = fields_vec[2];
  cerr << "cid1 filed=" << cid1 << ", cid2 field=" << cid2 << endl;
  string c1_lower = "1319";
  string c1_upper = "6196";
  string c2_lower = "1620";
  string c2_upper = "6197";
  RangeFilter *c1_range_filter =
      MakeRangeFilter(StringToByteArray(cid1), StringToByteArray(c1_lower),
                      StringToByteArray(c1_upper), false, true);
  SetRangeFilter(range_filters, 0, c1_range_filter);
  RangeFilter *c2_range_filter =
      MakeRangeFilter(StringToByteArray(cid2), StringToByteArray(c2_lower),
                      StringToByteArray(c2_upper), false, true);
  SetRangeFilter(range_filters, 1, c2_range_filter);

  // request number = 1
  struct Request *request =
      MakeRequest(10, vector_querys, 1, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                  0, nullptr, TRUE, 0);

  struct Response *response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(1.0f, response->results[0]->result_items[0]->score);
  DestroyResponse(response);

  // request number = 1, range filter number = 2
  request = MakeRequest(10, vector_querys, 1, nullptr, 0, range_filters, 2,
                        nullptr, 0, 1, 0, nullptr, TRUE, 0);
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  SearchResult *result = response->results[0];
  ASSERT_GT(result->result_num, 0);
  for (int i = 0; i < result->result_num; i++) {
    ResultItem *result_item = result->result_items[i];
    Field *cid1_field = GetField(result_item->doc, 1);
    string cid1_name = ByteArrayToString(cid1_field->name);
    int cid1_value = ByteArrayToInt(cid1_field->value);
    ASSERT_EQ(fields_vec[1], cid1_name)
        << "i=" << i << ", cid1_name=" << cid1_name;
    ASSERT_GE(cid1_value, std::stoi(c1_lower)) << "i=" << i;
    ASSERT_LE(cid1_value, std::stoi(c1_upper)) << "i=" << i;

    Field *cid2_field = GetField(result_item->doc, 2);
    string cid2_name = ByteArrayToString(cid2_field->name);
    int cid2_value = ByteArrayToInt(cid2_field->value);
    ASSERT_EQ(fields_vec[2], cid2_name) << "i=" << i;
    ASSERT_GE(cid2_value, std::stoi(c2_lower)) << "i=" << i;
    ASSERT_LE(cid2_value, std::stoi(c2_upper)) << "i=" << i;
  }
  DestroyResponse(response);

  // request number = 2
  request = MakeRequest(10, vector_querys, 1, nullptr, 0, range_filters, 2,
                        nullptr, 0, 2, 0, nullptr, TRUE, 0);
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(2, response->req_num);
  ASSERT_EQ(1.0f, response->results[0]->result_items[0]->score);
  ASSERT_EQ(1.0f, response->results[1]->result_items[0]->score);
  DestroyResponse(response);

  Dump(engine);

  Close(engine);

  printf("Finshed!\n");
}

void TestOneUrl() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files";
  string log_dir = "log";
  int max_doc_size = 5000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  std::vector<string> fields_vec = {"sku", "_id", "cid1", "cid2", "cid3"};
  std::vector<enum DataType> fields_type = {LONG, STRING, INT, INT, INT};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  struct VectorInfo **vectors_info = MakeVectorInfos(1);
  string model_id = "model";
  string vector_name = "abc";
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  struct VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(vector_name), FLOAT, TRUE, d,
      StringToByteArray(model_id), StringToByteArray(retrieval_type),
      StringToByteArray(store_type), nullptr);
  SetVectorInfo(vectors_info, 0, vector_info);

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, 1, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();

  string profile_file = "/root/wxd/feat_dir/sku_url_cid0_1.txt";
  string feature_file = "/root/wxd/feat_dir/feat_same0_0.dat";

  FILE *fp_feature = fopen(feature_file.c_str(), "rb");
  EXPECT_NE(fp_feature, nullptr);

  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> xb(d * 1);
  std::vector<float> search_feat(d * 1);
  long search_doc_id = 4;  // small
  // long search_doc_id = 13;
  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 1);
    auto profiles = std::move(utils::split(str, "\t"));

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field = MakeField(name, value, NULL, data_type);
      SetField(fields, i, field);
    }

    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *value = FloatToByteArray(xb.data(), d);
    struct ByteArray *name = StringToByteArray(vector_name);
    struct Field *field =
        MakeField(name, value, StringToByteArray(profiles[1]), VECTOR);
    SetField(fields, fields_vec.size(), field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 1);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    if (idx == search_doc_id) {
      search_feat.assign(xb.begin(), xb.end());
    }

    ++idx;
    if (idx > 70000) {
      break;
    }
  }

  std::thread t(BuildIndex, engine);
  t.detach();

  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 1);
    auto profiles = std::move(utils::split(str, "\t"));

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field = MakeField(name, value, NULL, data_type);
      SetField(fields, i, field);
    }

    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *value = FloatToByteArray(xb.data(), d);
    struct ByteArray *name = StringToByteArray(vector_name);
    struct Field *field =
        MakeField(name, value, StringToByteArray(profiles[1]), VECTOR);
    SetField(fields, fields_vec.size(), field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 1);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    ++idx;
    if (idx > 2000000) {
      break;
    }
  }

  fin.close();
  fclose(fp_feature);

  double add_time = utils::getmillisecs() - start;

  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  cerr << "waiting 1 seconds, then to search.....";
  std::this_thread::sleep_for(std::chrono::seconds(1));

  struct VectorQuery **vector_querys = MakeVectorQuerys(1);
  struct ByteArray *value;

  ASSERT_EQ(d, search_feat.size());
  value = FloatToByteArray(search_feat.data(), d);

  struct VectorQuery *vector_query = MakeVectorQuery(
      StringToByteArray(vector_name), value, 0, 1000000, 0.1, 0);
  SetVectorQuery(vector_querys, 0, vector_query);

  RangeFilter **range_filters = MakeRangeFilters(1);
  string cid1 = fields_vec[2];
  string cid2 = fields_vec[3];
  string cid3 = fields_vec[4];
  cerr << "cid1 filed=" << cid1 << ", cid2 field=" << cid2
       << ", cid3 field=" << cid3 << endl;
  string c3_lower = "1371";  // 1349, 1371_small
  string c3_upper = "1371";  // 1349, 1371
  RangeFilter *range_filter =
      MakeRangeFilter(StringToByteArray(cid3), StringToByteArray(c3_lower),
                      StringToByteArray(c3_upper), false, true);
  RangeFilter *range_filter1 =
      MakeRangeFilter(StringToByteArray(cid2), StringToByteArray("1342"),
                      StringToByteArray("1342"), false, true);
  SetRangeFilter(range_filters, 0, range_filter);
  SetRangeFilter(range_filters, 1, range_filter1);

  struct Request *request =
      MakeRequest(10, vector_querys, 1, nullptr, 0, range_filters, 1, nullptr,
                  0, 1, 0, nullptr, TRUE, 0);
  // range_filters, 1, nullptr, 0, 1);

  start = utils::getmillisecs();
  int search_num = 10000;
  for (int i = 0; i < search_num; i++) {
    struct Response *response = Search(engine, request);
    /*cerr << "response req_num=" << response->req_num << endl;
    SearchResult *search_result = GetSearchResult(response, 0);
    for (int i = 0; i < search_result->result_num; ++i) {
      struct ResultItem *result_item = GetResultItem(search_result, i);
      printf("i=%d, score [%f],  ", i, result_item->score);
      string msg = "";
      printDoc(result_item->doc, msg);
      printf("%s \n\n", msg.c_str());
    }
    */
    // PrintResponse(response);
    DestroyResponse(response);
  }
  double end = utils::getmillisecs();
  cerr << "test one url total cost=" << end - start
       << "ms, search num=" << search_num << endl;
  request->topn = 1000;
  struct Response *response = Search(engine, request);
  PrintResponse(response);

  Dump(engine);

  Close(engine);

  printf("Finshed!\n");
}

void TestSearchDirectlyL2() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "TestSearchDirectlyL2_files";
  string log_dir = "TestSearchDirectlyL2_log";
  int max_doc_size = 5000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  std::vector<string> fields_vec = {"sku", "_id", "cid1", "cid2", "cid3"};
  std::vector<enum DataType> fields_type = {LONG, STRING, INT, INT, INT};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  struct VectorInfo **vectors_info = MakeVectorInfos(1);
  string model_id = "model";
  string vector_name = "abc";
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  struct VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(vector_name), FLOAT, TRUE, d,
      StringToByteArray(model_id), StringToByteArray(retrieval_type),
      StringToByteArray(store_type), nullptr);
  SetVectorInfo(vectors_info, 0, vector_info);

  IVFPQParameters *pqParam = MakeIVFPQParameters(L2, 50, 256, 32, 8);
  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, 1, pqParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();

  string profile_file = "/root/wxd/feat_dir/sku_url_cid0_1.txt";
  string feature_file = "/root/wxd/feat_dir/feat_same0_0.dat";

  FILE *fp_feature = fopen(feature_file.c_str(), "rb");
  EXPECT_NE(fp_feature, nullptr);

  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> xb(d * 1);
  std::vector<float> search_feat(d * 1);
  long search_doc_id = 4;  // small
  // long search_doc_id = 13;
  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 1);
    auto profiles = std::move(utils::split(str, "\t"));

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field = MakeField(name, value, NULL, data_type);
      SetField(fields, i, field);
    }

    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *value = FloatToByteArray(xb.data(), d);
    struct ByteArray *name = StringToByteArray(vector_name);
    struct Field *field =
        MakeField(name, value, StringToByteArray(profiles[1]), VECTOR);
    SetField(fields, fields_vec.size(), field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 1);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    if (idx == search_doc_id) {
      search_feat.assign(xb.begin(), xb.end());
    }

    ++idx;
    if (idx > 3) {
      break;
    }
  }

  fin.close();
  fclose(fp_feature);

  int fd = open(feature_file.c_str(), O_RDONLY, 0);
  size_t mmap_size = 100 * sizeof(float) * d;
  float *all_feature =
      static_cast<float *>(mmap(NULL, mmap_size, PROT_READ, MAP_SHARED, fd, 0));
  close(fd);

  double add_time = utils::getmillisecs() - start;

  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  struct VectorQuery **vector_querys = MakeVectorQuerys(1);
  struct ByteArray *value;

  value = FloatToByteArray(all_feature, d);

  struct VectorQuery *vector_query = MakeVectorQuery(
      StringToByteArray(vector_name), value, 0, 1000000, 0.1, 0);
  SetVectorQuery(vector_querys, 0, vector_query);

  struct Request *request =
      MakeRequest(10, vector_querys, 1, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                  0, nullptr, TRUE, 0);
  request->topn = 100;
  struct Response *response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(4, response->results[0]->result_num);
  ASSERT_EQ(0, response->results[0]->result_items[0]->score);
  DestroyResponse(response);

  // search after delete 0
  string del_docid =
      "jfs/t154/294/2731244841/132212/aeedae14/53d72ee8Nfb5d326d.jpg";
  assert(0 == DelDoc(engine, StringToByteArray(del_docid)));
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(3, response->results[0]->result_num);
  DestroyResponse(response);

  // search after delete 2
  del_docid = "jfs/t2890/57/1536529718/401130/5156ac6e/57427c3fN3c367faf.jpg";
  assert(0 == DelDoc(engine, StringToByteArray(del_docid)));
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(2, response->results[0]->result_num);
  DestroyResponse(response);

  munmap(all_feature, mmap_size);

  Dump(engine);

  Close(engine);

  printf("Finshed!\n");
}

void TestSearchWithoutVector() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "TestSearchWithoutVector_files";
  string log_dir = "TestSearchWithoutVector_log";
  int max_doc_size = 5000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  std::vector<string> fields_vec = {"sku", "_id", "cid1", "cid2", "cid3"};
  std::vector<enum DataType> fields_type = {LONG, STRING, INT, INT, INT};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  struct VectorInfo **vectors_info = MakeVectorInfos(1);
  string model_id = "model";
  string vector_name = "abc";
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  struct VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(vector_name), FLOAT, TRUE, d,
      StringToByteArray(model_id), StringToByteArray(retrieval_type),
      StringToByteArray(store_type), nullptr);
  SetVectorInfo(vectors_info, 0, vector_info);

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, 1, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();

  string profile_file = "/root/wxd/feat_dir/sku_url_cid0_1.txt";
  string feature_file = "/root/wxd/feat_dir/feat_same0_0.dat";

  FILE *fp_feature = fopen(feature_file.c_str(), "rb");
  EXPECT_NE(fp_feature, nullptr);

  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> xb(d * 1);
  std::vector<float> search_feat(d * 1);
  long search_doc_id = 4;
  // long search_doc_id = 13;
  int total_send_num = 70000;
  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 1);
    auto profiles = std::move(utils::split(str, "\t"));

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field = MakeField(name, value, NULL, data_type);
      SetField(fields, i, field);
    }

    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *value = FloatToByteArray(xb.data(), d);
    struct ByteArray *name = StringToByteArray(vector_name);
    struct Field *field =
        MakeField(name, value, StringToByteArray(profiles[1]), VECTOR);
    SetField(fields, fields_vec.size(), field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 1);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    if (idx == search_doc_id) {
      search_feat.assign(xb.begin(), xb.end());
    }

    ++idx;
    if (idx >= total_send_num) {
      break;
    }
  }

  std::thread t(BuildIndex, engine);
  t.detach();

  fin.close();
  fclose(fp_feature);

  double add_time = utils::getmillisecs() - start;

  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  cerr << "waiting 1 seconds, then to search.....";
  std::this_thread::sleep_for(std::chrono::seconds(1));

  RangeFilter **range_filters = MakeRangeFilters(1);
  string cid1 = fields_vec[2];
  string cid2 = fields_vec[3];
  string cid3 = fields_vec[4];
  cerr << "cid1 filed=" << cid1 << ", cid2 field=" << cid2
       << ", cid3 field=" << cid3 << endl;
  RangeFilter *range_filter =
      MakeRangeFilter(StringToByteArray(cid3), StringToByteArray("1371"),
                      StringToByteArray("1371"), false, true);
  RangeFilter *range_filter1 =
      MakeRangeFilter(StringToByteArray(cid2), StringToByteArray("1345"),
                      StringToByteArray("1345"), false, true);
  SetRangeFilter(range_filters, 0, range_filter);
  SetRangeFilter(range_filters, 1, range_filter1);

  // search without range filter
  int topn = 10;
  struct Request *request = MakeRequest(topn, nullptr, 0, nullptr, 0, nullptr,
                                        0, nullptr, 0, 1, 0, nullptr, TRUE, 0);
  struct Response *response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(total_send_num, response->results[0]->total);
  ASSERT_EQ(topn, response->results[0]->result_num);
  DestroyResponse(response);

  // search with range filter
  request->range_filters = range_filters;
  request->range_filters_num = 2;
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(1001, response->results[0]->total);
  ASSERT_EQ(topn, response->results[0]->result_num);
  DestroyResponse(response);

  Dump(engine);

  Close(engine);

  printf("Finshed!\n");
}

void TestDelDocByQuery() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "TestDelDocByQuery_files";
  string log_dir = "TestDelDocByQuery_log";
  int max_doc_size = 5000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  std::vector<string> fields_vec = {"sku", "_id", "cid1", "cid2", "cid3"};
  std::vector<enum DataType> fields_type = {LONG, STRING, INT, INT, INT};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  struct VectorInfo **vectors_info = MakeVectorInfos(1);
  string model_id = "model";
  string vector_name = "abc";
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  struct VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(vector_name), FLOAT, TRUE, d,
      StringToByteArray(model_id), StringToByteArray(retrieval_type),
      StringToByteArray(store_type), nullptr);
  SetVectorInfo(vectors_info, 0, vector_info);

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, 1, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();

  string profile_file = "/root/wxd/feat_dir/sku_url_cid0_1.txt";
  string feature_file = "/root/wxd/feat_dir/feat_same0_0.dat";

  FILE *fp_feature = fopen(feature_file.c_str(), "rb");
  EXPECT_NE(fp_feature, nullptr);

  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> xb(d * 1);
  std::vector<float> search_feat(d * 1);
  long search_doc_id = 4;
  // long search_doc_id = 13;
  int total_send_num = 70000;
  std::vector<std::vector<string>> added_profiles;
  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + 1);
    auto profiles = std::move(utils::split(str, "\t"));
    added_profiles.push_back(profiles);

    for (size_t i = 0; i < fields_vec.size(); ++i) {
      enum DataType data_type = fields_type[i];
      struct ByteArray *name = StringToByteArray(fields_vec[i]);
      struct ByteArray *value;

      if (fields_type[i] == INT) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else if (fields_type[i] == LONG) {
        value =
            static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(profiles[i].c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(profiles[i]);
      }
      struct Field *field = MakeField(name, value, NULL, data_type);
      SetField(fields, i, field);
    }

    fread((void *)xb.data(), sizeof(float), d, fp_feature);
    struct ByteArray *value = FloatToByteArray(xb.data(), d);
    struct ByteArray *name = StringToByteArray(vector_name);
    struct Field *field =
        MakeField(name, value, StringToByteArray(profiles[1]), VECTOR);
    SetField(fields, fields_vec.size(), field);

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 1);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    if (idx == search_doc_id) {
      search_feat.assign(xb.begin(), xb.end());
    }

    ++idx;
    if (idx >= total_send_num) {
      break;
    }
  }

  std::thread t(BuildIndex, engine);
  t.detach();

  fin.close();
  fclose(fp_feature);

  double add_time = utils::getmillisecs() - start;

  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  cerr << "waiting 1 seconds, then to search.....";
  std::this_thread::sleep_for(std::chrono::seconds(1));

  RangeFilter **range_filters = MakeRangeFilters(1);
  string cid1 = fields_vec[2];
  string cid2 = fields_vec[3];
  string cid3 = fields_vec[4];
  cerr << "cid1 filed=" << cid1 << ", cid2 field=" << cid2
       << ", cid3 field=" << cid3 << endl;
  RangeFilter *range_filter =
      MakeRangeFilter(StringToByteArray(cid3), StringToByteArray("1371"),
                      StringToByteArray("1371"), false, true);
  RangeFilter *range_filter1 =
      MakeRangeFilter(StringToByteArray(cid2), StringToByteArray("1345"),
                      StringToByteArray("1345"), false, true);
  SetRangeFilter(range_filters, 0, range_filter);
  SetRangeFilter(range_filters, 1, range_filter1);

  // search without range filter
  int topn = 10;
  struct Request *request = MakeRequest(topn, nullptr, 0, nullptr, 0, nullptr,
                                        0, nullptr, 0, 1, 0, nullptr, TRUE, 0);
  struct Response *response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(total_send_num, response->results[0]->total);
  ASSERT_EQ(topn, response->results[0]->result_num);
  DestroyResponse(response);

  // search with range filter
  request->range_filters = range_filters;
  request->range_filters_num = 2;
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(1001, response->results[0]->total);
  ASSERT_EQ(topn, response->results[0]->result_num);
  DestroyResponse(response);

  // delete by range filter
  ASSERT_EQ(0, DelDocByQuery(engine, request));

  // search with range filter after delete by query
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(0, response->results[0]->total);
  ASSERT_EQ(0, response->results[0]->result_num);
  DestroyResponse(response);

  // search without range filter after delete by query
  request->range_filters = nullptr;
  request->range_filters_num = 0;
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  ASSERT_EQ(total_send_num - 1001, response->results[0]->total);
  ASSERT_EQ(topn, response->results[0]->result_num);
  DestroyResponse(response);

  int count = 0;
  for (size_t i = 0; i < added_profiles.size(); i++) {
    std::vector<string> profile = added_profiles[i];
    if (profile[4] == "1371") {
      ASSERT_EQ(nullptr,
                GetDocByID(engine, MakeByteArray(profile[1].c_str(),
                                                 profile[1].length())));
      count++;
    } else {
      ASSERT_NE(nullptr,
                GetDocByID(engine, MakeByteArray(profile[1].c_str(),
                                                 profile[1].length())));
    }
  }
  ASSERT_EQ(1001, count);

  Dump(engine);

  Close(engine);

  printf("Finshed!\n");
}

int FillFields(struct Field **fields, std::vector<string> fields_vec,
               std::vector<enum DataType> fields_type,
               std::vector<string> profiles) {
  for (size_t i = 0; i < fields_vec.size(); ++i) {
    enum DataType data_type = fields_type[i];
    struct ByteArray *name = StringToByteArray(fields_vec[i]);
    struct ByteArray *value;

    if (fields_type[i] == INT) {
      value = static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
      value->value = static_cast<char *>(malloc(sizeof(int)));
      value->len = sizeof(int);
      int v = atoi(profiles[i].c_str());
      memcpy(value->value, &v, value->len);
    } else if (fields_type[i] == LONG) {
      value = static_cast<struct ByteArray *>(malloc(sizeof(struct ByteArray)));
      value->value = static_cast<char *>(malloc(sizeof(long)));
      value->len = sizeof(long);
      long v = atol(profiles[i].c_str());
      memcpy(value->value, &v, value->len);
    } else {
      value = StringToByteArray(profiles[i]);
    }
    struct Field *field =
        MakeField(name, value, StringToByteArray(string("aaaaa")), data_type);
    SetField(fields, i, field);
  }

  return 0;
}

int FillVectorField(struct Field **fields, int id, string vector_name,
                    float *data, int d, string source) {
  struct ByteArray *name_ba = StringToByteArray(vector_name);
  struct ByteArray *value_ba = FloatToByteArray(data, d);
  struct ByteArray *source_ba = StringToByteArray(source);
  struct Field *field = MakeField(name_ba, value_ba, source_ba, VECTOR);
  SetField(fields, id, field);
  return 0;
}

string CreateVectorName(int i) { return "vector_" + std::to_string(i); }

void ReadFileOffset(std::ifstream &fs, long offset, char *output, long len) {
  long curr_offset = fs.tellg();
  fs.seekg(offset, std::ios_base::beg);
  cerr << "ReadFileOffset current offset=" << curr_offset
       << ", offset=" << offset << endl;
  fs.read(output, len);
}

void TestMultiIndex() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files_range_multi_index";
  string log_dir = "files_range_multi_index_log";
  int max_doc_size = 1000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  // _id = sku
  std::vector<string> fields_vec = {"_id"};
  std::vector<enum DataType> fields_type = {STRING};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  int vector_num = 2;
  struct VectorInfo **vectors_info = MakeVectorInfos(vector_num);
  string retrieval_type = "IVFPQ";
  string store_type = "Mmap";
  for (int i = 0; i < vector_num; i++) {
    string model_id = "model_" + std::to_string(i);
    string vector_name = CreateVectorName(i);
    struct VectorInfo *vector_info = MakeVectorInfo(
        StringToByteArray(vector_name), FLOAT, TRUE, d,
        StringToByteArray(model_id), StringToByteArray(retrieval_type),
        StringToByteArray(store_type), nullptr);
    SetVectorInfo(vectors_info, i, vector_info);
  }

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, vector_num, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();
  int vector_byte_size = sizeof(float) * d;
  string dir = "/root/cpp_dev/istore/gamma/src/searcher/build";

  string profile_file = dir + "/output_url_1w.txt";
  string qian_feature_file = dir + "/output_feat_qian_1w.dat";
  string hou_feature_file = dir + "/output_feat_hou_1w.dat";

  std::ifstream qian_feature_fp(qian_feature_file.c_str(),
                                std::ios_base::binary);
  std::ifstream hou_feature_fp(hou_feature_file.c_str(), std::ios_base::binary);
  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> qian_xb(d * 1);
  std::vector<float> hou_xb(d * 1);
  int full_doc_num = 70000;

  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + vector_num);
    std::vector<string> profiles = utils::split(str, " ");
    if (profiles.size() < 1) {
      cerr << "idx=" << idx << "invalid split size=" << profiles.size()
           << ", line=" << str << endl;
      break;
    }
    string url = profiles[0];
    if (url == "") {
      cerr << "idx=" << idx << "invalid url, url=" << url << endl;
      break;
    }

    ASSERT_EQ(0, FillFields(fields, fields_vec, fields_type, profiles))
        << "fill fields error";
    qian_feature_fp.read((char *)qian_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size(), CreateVectorName(0),
                                 qian_xb.data(), d, url));
    hou_feature_fp.read((char *)hou_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size() + 1,
                                 CreateVectorName(1), hou_xb.data(), d, url));

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 2);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    ++idx;
    if (idx >= full_doc_num) {
      break;
    }
  }
  // fin.close();

  double add_time = utils::getmillisecs() - start;
  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  std::thread t(BuildIndex, engine);
  t.detach();

  cerr << "waiting to build....." << endl;
  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  int query_doc_id = 2999;
  string url =
      "jfs/t1/17842/6/2563/424915/5c1e1475Ed3c1c0b8/9cba1ea2f14082e5.jpg";
  size_t offset = vector_byte_size * query_doc_id;
  ReadFileOffset(qian_feature_fp, offset, (char *)qian_xb.data(),
                 vector_byte_size);
  ReadFileOffset(hou_feature_fp, offset, (char *)hou_xb.data(),
                 vector_byte_size);

  int vector_query_num = 2;
  struct VectorQuery **vector_querys = MakeVectorQuerys(vector_query_num);
  struct ByteArray *value = FloatToByteArray(qian_xb.data(), d);
  struct VectorQuery *vector_query0 = MakeVectorQuery(
      StringToByteArray(CreateVectorName(0)), value, 0, 0.5, 2, 1);
  value = FloatToByteArray(hou_xb.data(), d);
  struct VectorQuery *vector_query1 = MakeVectorQuery(
      StringToByteArray(CreateVectorName(1)), value, 0, 0.5, 2, 1);

  RangeFilter **range_filters = MakeRangeFilters(1);
  ByteArray *cid3_ba = StringToByteArray(string("cid3"));
  string c3_lower = "1320";
  string c3_upper = "1320";
  RangeFilter *range_filter =
      MakeRangeFilter(cid3_ba, StringToByteArray(c3_lower),
                      StringToByteArray(c3_upper), true, true);
  SetRangeFilter(range_filters, 0, range_filter);

  // request by two vector
  SetVectorQuery(vector_querys, 0, vector_query0);
  SetVectorQuery(vector_querys, 1, vector_query1);
  Request *request =
      MakeRequest(10, vector_querys, vector_query_num, nullptr, 0, nullptr, 0,
                  nullptr, 0, 1, 0, nullptr, TRUE, 0);

  Response *response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  struct SearchResult *search_result = response->results[0];
  ASSERT_EQ(5, search_result->result_num);
  for (int i = 0; i < search_result->result_num; ++i) {
    struct ResultItem *result_item = GetResultItem(search_result, i);
    cJSON *extra = cJSON_Parse(ByteArrayToString(result_item->extra).c_str());
    ASSERT_NE(nullptr, extra) << "i=" << i;
    cJSON *vec_result = cJSON_GetObjectItem(extra, "vector_result");
    ASSERT_EQ(2, cJSON_GetArraySize(vec_result));
    cJSON *vec0 = cJSON_GetArrayItem(vec_result, 0);
    cJSON *vec1 = cJSON_GetArrayItem(vec_result, 1);
    cJSON *score0 = cJSON_GetObjectItem(vec0, "score");
    cJSON *score1 = cJSON_GetObjectItem(vec1, "score");
    ASSERT_LE(vector_query0->min_score * 2, score0->valuedouble);
    ASSERT_GE(vector_query0->max_score * 2, score0->valuedouble);
    ASSERT_LE(vector_query1->min_score * 2, score1->valuedouble);
    ASSERT_GE(vector_query1->max_score * 2, score1->valuedouble);
  }
  DestroyResponse(response);

  // request by limited score
  vector_query0->min_score = 0.471754f;
  vector_query1->max_score = 0.499883f;
  response = Search(engine, request);
  PrintResponse(response);
  ASSERT_EQ(1, response->req_num);
  search_result = response->results[0];
  ASSERT_EQ(2, search_result->result_num);
  for (int i = 0; i < search_result->result_num; ++i) {
    struct ResultItem *result_item = GetResultItem(search_result, i);
    cJSON *extra = cJSON_Parse(ByteArrayToString(result_item->extra).c_str());
    ASSERT_NE(nullptr, extra) << "i=" << i;
    cJSON *vec_result = cJSON_GetObjectItem(extra, "vector_result");
    ASSERT_EQ(2, cJSON_GetArraySize(vec_result));
    cJSON *vec0 = cJSON_GetArrayItem(vec_result, 0);
    cJSON *vec1 = cJSON_GetArrayItem(vec_result, 1);
    cJSON *score0 = cJSON_GetObjectItem(vec0, "score");
    cJSON *score1 = cJSON_GetObjectItem(vec1, "score");
    ASSERT_LE(vector_query0->min_score * 2, score0->valuedouble);
    ASSERT_GE(vector_query0->max_score * 2, score0->valuedouble);
    ASSERT_LE(vector_query1->min_score * 2, score1->valuedouble);
    ASSERT_GE(vector_query1->max_score * 2, score1->valuedouble);
  }
  DestroyResponse(response);

  // reset score
  vector_query0->min_score = 0;
  vector_query1->max_score = 1;

  // delete doc
  ASSERT_EQ(0, DelDoc(engine, StringToByteArray(url)));
  response = Search(engine, request);
  cerr << "####search after delete" << endl;
  PrintResponse(response);
  DestroyResponse(response);

  struct Field **fields = MakeFields(fields_vec.size() + vector_num);
  std::vector<string> profiles = {"re_add_0"};

  ASSERT_EQ(0, FillFields(fields, fields_vec, fields_type, profiles))
      << "fill fields error";
  ASSERT_EQ(0, FillVectorField(fields, fields_vec.size(), CreateVectorName(0),
                               qian_xb.data(), d, url));
  ASSERT_EQ(0, FillVectorField(fields, fields_vec.size() + 1,
                               CreateVectorName(1), hou_xb.data(), d, url));
  struct Doc *doc = MakeDoc(fields, fields_vec.size() + 2);
  ASSERT_EQ(0, AddDoc(engine, doc));
  DestroyDoc(doc);
  std::this_thread::sleep_for(std::chrono::seconds(10));
  response = Search(engine, request);
  cerr << "####search after re-add" << endl;
  PrintResponse(response);
  DestroyResponse(response);

  Dump(engine);

  Close(engine);

  printf("TestMultiIndex Finshed!\n");
}

void TestMultiIndexResultConsistent() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files_range_multi_index";
  string log_dir = "files_range_multi_index_log";
  int max_doc_size = 1000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  // _id = sku
  std::vector<string> fields_vec = {"_id"};
  std::vector<enum DataType> fields_type = {STRING};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  int vector_num = 2;
  struct VectorInfo **vectors_info = MakeVectorInfos(vector_num);
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  for (int i = 0; i < vector_num; i++) {
    string model_id = "model_" + std::to_string(i);
    string vector_name = CreateVectorName(i);
    struct VectorInfo *vector_info = MakeVectorInfo(
        StringToByteArray(vector_name), FLOAT, TRUE, d,
        StringToByteArray(model_id), StringToByteArray(retrieval_type),
        StringToByteArray(store_type), nullptr);
    SetVectorInfo(vectors_info, i, vector_info);
  }

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, vector_num, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();
  int vector_byte_size = sizeof(float) * d;

  string profile_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_url.txt";
  string qian_feature_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_feat_qian.dat";
  string hou_feature_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_feat_hou.dat";

  std::ifstream qian_feature_fp(qian_feature_file.c_str(),
                                std::ios_base::binary);
  std::ifstream hou_feature_fp(hou_feature_file.c_str(), std::ios_base::binary);
  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> qian_xb(d * 1);
  std::vector<float> hou_xb(d * 1);
  int full_doc_num = 70000;

  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + vector_num);
    std::vector<string> profiles = utils::split(str, " ");
    if (profiles.size() < 1) {
      cerr << "idx=" << idx << "invalid split size=" << profiles.size()
           << ", line=" << str << endl;
      break;
    }
    string url = profiles[0];
    if (url == "") {
      cerr << "idx=" << idx << "invalid url, url=" << url << endl;
      break;
    }

    ASSERT_EQ(0, FillFields(fields, fields_vec, fields_type, profiles))
        << "fill fields error";
    qian_feature_fp.read((char *)qian_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size(), CreateVectorName(0),
                                 qian_xb.data(), d, url));
    hou_feature_fp.read((char *)hou_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size() + 1,
                                 CreateVectorName(1), hou_xb.data(), d, url));

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 2);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    ++idx;
    if (idx >= full_doc_num) {
      break;
    }
  }
  // fin.close();

  double add_time = utils::getmillisecs() - start;
  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  std::thread t(BuildIndex, engine);
  t.detach();

  cerr << "waiting to build....." << endl;
  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  int search_num = 1000;
  std::srand(std::time(nullptr));
  int search_idxs[search_num];
  for (int i = 0; i < search_num; i++) {
    search_idxs[i] = std::rand() % full_doc_num;
  }

  for (int n = 0; n < search_num; n++) {
    int query_doc_id = search_idxs[n];
    cerr << "++++++++++++++!!!!!!!!!!!!!!!!!!&&&&&&&&&&&&&& n=" << n
         << ", doc id=" << query_doc_id << endl;
    string url =
        "jfs/t1/17842/6/2563/424915/5c1e1475Ed3c1c0b8/9cba1ea2f14082e5.jpg";
    size_t offset = vector_byte_size * query_doc_id;
    ReadFileOffset(qian_feature_fp, offset, (char *)qian_xb.data(),
                   vector_byte_size);
    ReadFileOffset(hou_feature_fp, offset, (char *)hou_xb.data(),
                   vector_byte_size);

    int vector_query_num = 2;
    struct VectorQuery **vector_querys = MakeVectorQuerys(vector_query_num);
    struct ByteArray *value = FloatToByteArray(qian_xb.data(), d);
    struct VectorQuery *vector_query0 = MakeVectorQuery(
        StringToByteArray(CreateVectorName(0)), value, 0.1, 1, 2, 1);
    value = FloatToByteArray(hou_xb.data(), d);
    struct VectorQuery *vector_query1 = MakeVectorQuery(
        StringToByteArray(CreateVectorName(1)), value, 0.1, 1, 2, 1);

    // request by vector 0
    SetVectorQuery(vector_querys, 0, vector_query0);
    struct Request *request =
        MakeRequest(10, vector_querys, 1, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                    0, nullptr, TRUE, 0);
    struct Response *response = Search(engine, request);
    PrintResponse(response);
    ASSERT_EQ(1, response->req_num);
    vector<string> doc_id_list0;
    GetResponseDocIds(response, 0, doc_id_list0);
    cerr << "########## request by vector0: doc id list="
         << VectorToString(doc_id_list0.data(), doc_id_list0.size())
         << ", size=" << doc_id_list0.size() << endl;
    map<string, bool> doc_id_map0;
    for (size_t i = 0; i < doc_id_list0.size(); i++) {
      doc_id_map0.insert(std::make_pair(doc_id_list0[i], true));
    }
    DestroyResponse(response);

    // request by vector 1
    SetVectorQuery(vector_querys, 0, vector_query1);
    request = MakeRequest(10, vector_querys, 1, nullptr, 0, nullptr, 0, nullptr,
                          0, 1, 0, nullptr, TRUE, 0);

    response = Search(engine, request);
    PrintResponse(response);
    ASSERT_EQ(1, response->req_num);
    vector<string> doc_id_list1;
    GetResponseDocIds(response, 0, doc_id_list1);
    cerr << "########## request by vector1: doc id list="
         << VectorToString(doc_id_list1.data(), doc_id_list1.size()) << endl;
    std::map<string, bool> comm_doc_id_map;
    for (size_t i = 0; i < doc_id_list1.size(); i++) {
      string doc_id = doc_id_list1[i];
      if (doc_id_map0.find(doc_id) != doc_id_map0.end()) {
        comm_doc_id_map.insert(std::make_pair(doc_id, true));
      }
    }
    DestroyResponse(response);
    cerr << "###### common doc id list=" << MapToString(comm_doc_id_map)
         << endl;

    // request by two vector
    SetVectorQuery(vector_querys, 0, vector_query0);
    SetVectorQuery(vector_querys, 1, vector_query1);
    request = MakeRequest(10, vector_querys, vector_query_num, nullptr, 0,
                          nullptr, 0, nullptr, 0, 1, 0, nullptr, TRUE, 0);

    response = Search(engine, request);
    PrintResponse(response);
    ASSERT_EQ(1, response->req_num);
    vector<string> doc_id_list;
    GetResponseDocIds(response, 0, doc_id_list);
    cerr << "########## request by two vector: doc id list="
         << VectorToString(doc_id_list.data(), doc_id_list.size()) << endl;
    ASSERT_EQ(comm_doc_id_map.size(), doc_id_list.size());
    for (size_t i = 0; i < doc_id_list.size(); i++) {
      ASSERT_NE(comm_doc_id_map.end(), comm_doc_id_map.find(doc_id_list[i]))
          << "i=" << i;
    }
    DestroyResponse(response);
  }

  Dump(engine);

  Close(engine);

  printf("TestMultiIndex Finshed!\n");
}

void TestMultiIndexSearchPerf() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files_range_multi_index_search_perf";
  string log_dir = "files_range_multi_index_search_perf_log";
  int max_doc_size = 1000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);

  EXPECT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  // _id = sku
  std::vector<string> fields_vec = {"_id"};
  std::vector<enum DataType> fields_type = {STRING};

  struct FieldInfo **field_infos = MakeFieldInfos(fields_vec.size());

  for (size_t i = 0; i < fields_vec.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(fields_vec[i]), fields_type[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  int vector_num = 2;
  struct VectorInfo **vectors_info = MakeVectorInfos(vector_num);
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  for (int i = 0; i < vector_num; i++) {
    string model_id = "model_" + std::to_string(i);
    string vector_name = CreateVectorName(i);
    struct VectorInfo *vector_info = MakeVectorInfo(
        StringToByteArray(vector_name), FLOAT, TRUE, d,
        StringToByteArray(model_id), StringToByteArray(retrieval_type),
        StringToByteArray(store_type), nullptr);
    SetVectorInfo(vectors_info, i, vector_info);
  }

  struct Table *table = MakeTable(table_name, field_infos, fields_vec.size(),
                                  vectors_info, vector_num, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  double start = utils::getmillisecs();
  int vector_byte_size = sizeof(float) * d;

  string profile_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_url.txt";
  string qian_feature_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_feat_qian.dat";
  string hou_feature_file =
      "/root/xiedabin/gamma_test_data/multi_vector_index/output_feat_hou.dat";

  std::ifstream qian_feature_fp(qian_feature_file.c_str(),
                                std::ios_base::binary);
  std::ifstream hou_feature_fp(hou_feature_file.c_str(), std::ios_base::binary);
  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long idx = 0;
  std::vector<float> qian_xb(d * 1);
  std::vector<float> hou_xb(d * 1);
  int full_doc_num = 70000;

  while (!fin.eof()) {
    std::getline(fin, str);
    if (str == "") break;

    struct Field **fields = MakeFields(fields_vec.size() + vector_num);
    std::vector<string> profiles = utils::split(str, " ");
    if (profiles.size() < 1) {
      cerr << "idx=" << idx << "invalid split size=" << profiles.size()
           << ", line=" << str << endl;
      break;
    }
    string url = profiles[0];
    if (url == "") {
      cerr << "idx=" << idx << "invalid url, url=" << url << endl;
      break;
    }

    ASSERT_EQ(0, FillFields(fields, fields_vec, fields_type, profiles))
        << "fill fields error";
    qian_feature_fp.read((char *)qian_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size(), CreateVectorName(0),
                                 qian_xb.data(), d, url));
    hou_feature_fp.read((char *)hou_xb.data(), vector_byte_size);
    ASSERT_EQ(0, FillVectorField(fields, fields_vec.size() + 1,
                                 CreateVectorName(1), hou_xb.data(), d, url));

    struct Doc *doc = MakeDoc(fields, fields_vec.size() + 2);
    AddDoc(engine, doc);
    DestroyDoc(doc);

    ++idx;
    if (idx >= full_doc_num) {
      break;
    }
  }
  // fin.close();

  double add_time = utils::getmillisecs() - start;
  printf("Add use time [%.1f]ms, num=%ld\n", add_time, idx);

  std::thread t(BuildIndex, engine);
  t.detach();

  cerr << "waiting to build....." << endl;
  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  int query_doc_id = 0;
  size_t offset = vector_byte_size * query_doc_id;
  ReadFileOffset(qian_feature_fp, offset, (char *)qian_xb.data(),
                 vector_byte_size);
  ReadFileOffset(hou_feature_fp, offset, (char *)hou_xb.data(),
                 vector_byte_size);

  int vector_query_num = 2;
  struct VectorQuery **vector_querys = MakeVectorQuerys(vector_query_num);
  struct ByteArray *value = FloatToByteArray(qian_xb.data(), d);
  struct VectorQuery *vector_query0 = MakeVectorQuery(
      StringToByteArray(CreateVectorName(0)), value, 0.1, 1, 2, 1);
  value = FloatToByteArray(hou_xb.data(), d);
  struct VectorQuery *vector_query1 = MakeVectorQuery(
      StringToByteArray(CreateVectorName(1)), value, 0.1, 1, 2, 1);

  // request by vector 0
  SetVectorQuery(vector_querys, 0, vector_query0);
  struct Request *request =
      MakeRequest(10, vector_querys, 1, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                  0, nullptr, TRUE, 0);

  int search_num = 10000;
  double sstart = utils::getmillisecs();
  for (int n = 0; n < search_num; n++) {
    struct Response *response = Search(engine, request);
    // PrintResponse(response);
    DestroyResponse(response);
  }
  double v0_send = utils::getmillisecs();

  // request by two vector
  SetVectorQuery(vector_querys, 0, vector_query0);
  SetVectorQuery(vector_querys, 1, vector_query1);
  request = MakeRequest(10, vector_querys, vector_query_num, nullptr, 0,
                        nullptr, 0, nullptr, 0, 1, 0, nullptr, TRUE, 0);
  for (int n = 0; n < search_num; n++) {
    struct Response *response = Search(engine, request);
    // PrintResponse(response);
    DestroyResponse(response);
  }
  double send = utils::getmillisecs();

  double v0_cost = v0_send - sstart;
  double v_cost = send - v0_send;
  cerr << "search finished, number=" << search_num
       << ", total cost=" << send - sstart
       << "ms, qeury one vector cost=" << v0_cost
       << ", avg latency=" << v0_cost / search_num
       << "ms, query two vector cost=" << v_cost
       << "ms, avg latency=" << v_cost / search_num << endl;

  Dump(engine);

  Close(engine);

  printf("TestMultiIndexSearchPerf Finshed!\n");
}

Doc *MakeAndFillDoc(vector<string> field_names,
                    vector<enum DataType> field_types,
                    vector<string> field_values, vector<string> vec_field_names,
                    vector<float *> vec_field_values, int dimension,
                    vector<string> vec_field_sources) {
  int field_num = field_names.size() + vec_field_names.size();
  Field **fields = MakeFields(field_num);

  for (size_t i = 0; i < field_names.size(); ++i) {
    enum DataType data_type = field_types[i];
    string field_value = field_values[i];
    ByteArray *name = StringToByteArray(field_names[i]);
    ByteArray *value = static_cast<ByteArray *>(malloc(sizeof(ByteArray)));
    if (data_type == INT) {
      value->value = static_cast<char *>(malloc(sizeof(int)));
      value->len = sizeof(int);
      int v = atoi(field_value.c_str());
      memcpy(value->value, &v, value->len);
    } else if (data_type == LONG) {
      value->value = static_cast<char *>(malloc(sizeof(long)));
      value->len = sizeof(long);
      long v = atol(field_value.c_str());
      memcpy(value->value, &v, value->len);
    } else {
      value = StringToByteArray(field_value);
    }
    ByteArray *source = StringToByteArray(string("aaa"));
    Field *field = MakeField(name, value, source, data_type);
    SetField(fields, i, field);
  }

  for (size_t j = 0; j < vec_field_names.size(); j++) {
    ByteArray *value = FloatToByteArray(vec_field_values[j], dimension);
    ByteArray *name = StringToByteArray(vec_field_names[j]);
    ByteArray *source = StringToByteArray(vec_field_sources[j]);
    Field *field = MakeField(name, value, source, VECTOR);
    SetField(fields, field_names.size() + j, field);
  }

  return MakeDoc(fields, field_num);
}

void TestGetDocAfterUpdate() {
  setvbuf(stdout, (char *)NULL, _IONBF, 0);
  string path = "files_range_get_doc_after_update";
  string log_dir = "files_range_get_doc_after_update_log";
  int max_doc_size = 1000000;
  struct Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  SetLogDictionary(StringToByteArray(log_dir));
  void *engine = Init(config);
  DestroyConfig(config);
  ASSERT_NE(engine, nullptr);

  struct ByteArray *table_name = MakeByteArray("test", 4);
  int d = 512;

  // _id = sku
  std::vector<string> field_names = {"_id", "price"};
  std::vector<enum DataType> field_types = {STRING, INT};
  std::vector<string> vec_field_names = {"feature"};

  struct FieldInfo **field_infos = MakeFieldInfos(field_names.size());

  for (size_t i = 0; i < field_names.size(); ++i) {
    struct FieldInfo *field_info =
        MakeFieldInfo(StringToByteArray(field_names[i]), field_types[i], 1);
    SetFieldInfo(field_infos, i, field_info);
  }

  int vector_num = vec_field_names.size();
  struct VectorInfo **vectors_info = MakeVectorInfos(vector_num);
  string retrieval_type = "IVFPQ";
  string store_type = "MemoryOnly";
  for (int i = 0; i < vector_num; i++) {
    string model_id = "model_" + std::to_string(i);
    string vector_name = vec_field_names[i];
    struct VectorInfo *vector_info = MakeVectorInfo(
        StringToByteArray(vector_name), FLOAT, TRUE, d,
        StringToByteArray(model_id), StringToByteArray(retrieval_type),
        StringToByteArray(store_type), nullptr);
    SetVectorInfo(vectors_info, i, vector_info);
  }

  struct Table *table = MakeTable(table_name, field_infos, field_names.size(),
                                  vectors_info, vector_num, kIVFPQParam);
  enum ResponseCode ret = CreateTable(engine, table);
  DestroyTable(table);

  printf("Create table ret [%d]\n", ret);

  string doc_key = "doc_1";
  vector<string> field_values = {doc_key, "10"};
  vector<float> xb(d);
  for (size_t i = 0; i < xb.size(); i++) {
    xb[i] = i;
  }
  vector<float *> vec_field_values = {xb.data()};
  vector<string> vec_field_sources = {"source_url"};
  Doc *doc =
      MakeAndFillDoc(field_names, field_types, field_values, vec_field_names,
                     vec_field_values, d, vec_field_sources);
  AddOrUpdateDoc(engine, doc);
  Doc *actual_doc = GetDocByID(engine, StringToByteArray(doc_key));
  ASSERT_NE(nullptr, actual_doc);
  string msg;
  printDoc(actual_doc, msg);
  cerr << "doc=" << msg << endl;
  DestroyDoc(doc);

  field_values[1] = "12";
  doc = MakeAndFillDoc(field_names, field_types, field_values, vec_field_names,
                       vec_field_values, d, vec_field_sources);
  AddOrUpdateDoc(engine, doc);
  actual_doc = GetDocByID(engine, StringToByteArray(doc_key));
  ASSERT_NE(nullptr, actual_doc);
  msg = "";
  printDoc(actual_doc, msg);
  cerr << "after update, doc=" << msg << endl;

  field_values[1] = "1212";
  doc = MakeAndFillDoc(field_names, field_types, field_values, vec_field_names,
                       vec_field_values, d, vec_field_sources);
  AddOrUpdateDoc(engine, doc);
  actual_doc = GetDocByID(engine, StringToByteArray(doc_key));
  ASSERT_NE(nullptr, actual_doc);
  msg = "";
  printDoc(actual_doc, msg);
  cerr << "after twice update, doc=" << msg << endl;

  DestroyDoc(doc);
}

void CheckVector() {
  std::ifstream fin("output_feat_qian.dat", std::ios_base::binary);
  int d = 512;
  int vector_byte_size = sizeof(float) * d;
  vector<float> xb(d);
  long idx = 0;
  while (!fin.eof()) {
    fin.read((char *)xb.data(), vector_byte_size);
    float score = 0.0f;
    for (int i = 0; i < d; i++) {
      score += xb[i] * xb[i];
    }
    cerr << "idx=" << idx++ << ", score=" << score << endl;
  }
}

void PrintUsage() {
  cerr << "Usage: test_files_range case_id" << endl;
  cerr << "Example: test_files_range 1" << endl;
  cerr << "\t 1:TestMultiUrl" << endl;
  cerr << "\t 2:TestOneUrl" << endl;
  cerr << "\t 3:TestMultiIndex" << endl;
  cerr << "\t 4:TestMultiIndexResultconsistent" << endl;
  cerr << "\t 5:TestMultiIndexSearchPerf" << endl;
  cerr << "\t 6:TestSearchWithoutVector" << endl;
  cerr << "\t 7:TestGetDocAfterUpdate" << endl;
  cerr << "\t 8:TestDelDocByQuery" << endl;
  cerr << "\t 9:TestSearchDirectlyL2" << endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    PrintUsage();
    return -1;
  }
  int case_id = std::stoi(argv[1]);
  cerr << "case id=" << case_id << endl;
  switch (case_id) {
    case 1:
      TestMultiUrl();
      break;
    case 2:
      TestOneUrl();
      break;
    case 3:
      TestMultiIndex();
      break;
    case 4:
      TestMultiIndexResultConsistent();
      break;
    case 5:
      TestMultiIndexSearchPerf();
      break;
    case 6:
      TestSearchWithoutVector();
      break;
    case 7:
      TestGetDocAfterUpdate();
      break;
    case 8:
      TestDelDocByQuery();
      break;
    case 9:
      TestSearchDirectlyL2();
      break;
    default:
      PrintUsage();
      return -1;
  }
  // TestMultiUrl();
  // TestOneUrl();
  // TestMultiIndex();
  // CheckVector();
  // TestGetDocAfterUpdate();
  return 0;
}
