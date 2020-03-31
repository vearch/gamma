/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <cmath>
#include <fstream>
#include <functional>
#include <future>
#include "test.h"

namespace Test {

struct Options {
  Options() {
    nprobe = 10;
    doc_id = 0;
    d = 512;
    max_doc_size = 10000 * 10;
    add_doc_num = 10000 * 1;
    search_num = 10000 * 3;
    fields_vec = {"sku", "_id", "cid1", "cid2", "cid3"};
    fields_type = {STRING, STRING, INT, INT, INT};
    vector_name = "abc";
    path = "files";
    string log_dir = "log";
    model_id = "model";
    retrieval_type = "IVFPQ";
    store_type = "Mmap";
    store_param = "{\"cache_size\": 256}";
    profiles.resize(search_num * fields_vec.size());
    feature = new float[d * search_num];
    engine = nullptr;
  }
  ~Options() {
    if (feature) {
      delete[] feature;
    }
  }

  int nprobe;
  int doc_id;
  int d;
  int max_doc_size;
  long add_doc_num;
  int search_num;
  std::vector<string> fields_vec;
  std::vector<enum DataType> fields_type;
  string path;
  string log_dir;
  string vector_name;
  string model_id;
  string retrieval_type;
  string store_type;
  string store_param;

  std::vector<string> profiles;
  float *feature;

  char *docids_bitmap_;
  void *engine;
};

static struct Options opt;

string profile_file =
    "./profile_10w.txt";
string feature_file =
    "./feat_10w.dat";

int AddDoc(void *engine, int start_id, int end_id, int interval = 0) {
  FILE *fet_fp = fopen(feature_file.c_str(), "rb");
  if (fet_fp == nullptr) {
    LOG(ERROR) << "open feature file error";
    return -1;
  }
  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;

  for (int i = 0; i < end_id; ++i) {
    double start = utils::getmillisecs();
    if (fin.eof()) {
      LOG(ERROR) << "profile is eof, i=" << i;
      return -1;
    }
    std::getline(fin, str);
    if (str == "") {
      LOG(ERROR) << "profile get empty line, i=" << i;
      return -1;
    }
    vector<string> profile = std::move(utils::split(str, "\t"));
    if (i < opt.search_num) {
      for (size_t j = 0; j < opt.fields_vec.size(); j++) {
        opt.profiles[i * opt.fields_vec.size() + j] = profile[j];
      }
    }
    float vector[opt.d];
    size_t ret = fread((void *)vector, sizeof(float), opt.d, fet_fp);
    assert(ret == (size_t)opt.d);
    if (i < opt.search_num) {
      memcpy((void *)(opt.feature + i * opt.d), (void *)vector,
             sizeof(float) * opt.d);
    }
    if (i < start_id) {
      continue;
    }

    Field **fields = MakeFields(opt.fields_vec.size() + 1);
    for (size_t j = 0; j < opt.fields_vec.size(); ++j) {
      enum DataType data_type = opt.fields_type[j];
      ByteArray *name = StringToByteArray(opt.fields_vec[j]);
      ByteArray *value;

      string &data = profile[j];
      if (opt.fields_type[j] == INT) {
        value = static_cast<ByteArray *>(malloc(sizeof(ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(int)));
        value->len = sizeof(int);
        int v = atoi(data.c_str());
        memcpy(value->value, &v, value->len);
      } else if (opt.fields_type[j] == LONG) {
        value = static_cast<ByteArray *>(malloc(sizeof(ByteArray)));
        value->value = static_cast<char *>(malloc(sizeof(long)));
        value->len = sizeof(long);
        long v = atol(data.c_str());
        memcpy(value->value, &v, value->len);
      } else {
        value = StringToByteArray(data);
      }
      ByteArray *source =
          StringToByteArray(string("jfs/t1/46413/10/6998/121644/"
                                   "5d493cfaE53b7c078/c4e2526e8f8a698f.jpg"));
      Field *field = MakeField(name, value, source, data_type);
      SetField(fields, j, field);
    }

    ByteArray *value = FloatToByteArray(vector, opt.d);
    ByteArray *name = StringToByteArray(opt.vector_name);
    ByteArray *source = StringToByteArray(string(
        "jfs/t1/46413/10/6998/121644/5d493cfaE53b7c078/c4e2526e8f8a698f.jpg"));
    Field *field = MakeField(name, value, source, VECTOR);
    SetField(fields, opt.fields_vec.size(), field);

    Doc *doc = MakeDoc(fields, opt.fields_vec.size() + 1);
    AddOrUpdateDoc(engine, doc);
    DestroyDoc(doc);
    ++opt.doc_id;
    double elap = utils::getmillisecs() - start;
    if (i % 10000 == 0) {
      LOG(INFO) << "AddDoc use [" << elap << "]ms";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
  }
  fin.close();
  fclose(fet_fp);
  return 0;
}

int SearchThread(void *engine, int num, int start_id) {
  int idx = 0;
  double time = 0;
  int failed_count = 0;
  int req_num = 1;
  string error;
  while (idx < num) {
    double start = utils::getmillisecs();
    VectorQuery **vector_querys = MakeVectorQuerys(1);
    int docid = start_id + idx;
    ByteArray *value = FloatToByteArray(opt.feature + (uint64_t)docid * opt.d,
                                        opt.d * req_num);
    VectorQuery *vector_query = MakeVectorQuery(
        StringToByteArray(opt.vector_name), value, 0, 10000, 0.1, 0);
    SetVectorQuery(vector_querys, 0, vector_query);

    string c3_lower = opt.profiles[docid * (opt.fields_vec.size()) + 4];
    Request *request = MakeRequest(100, vector_querys, 1, nullptr, 0, nullptr,
                                   0, nullptr, 0, req_num, 0, nullptr, TRUE, 0);

    Response *response = Search(engine, request);
    for (int i = 0; i < response->req_num; ++i) {
      int ii = docid + i;
      string msg = "docid=" + std::to_string(ii) + ", cid3=" + c3_lower + ", ";
      SearchResult *results = GetSearchResult(response, i);
      if (results->result_num <= 0) {
        if (failed_count < 1000) {
          LOG(INFO) << "result number <= 0, msg=" << msg
                    << ", code=" << results->result_code
                    << ", error=" << ByteArrayToString(results->msg);
          error += std::to_string(ii) + ",";
        }
        failed_count++;
        continue;
      }
      msg += string("total [") + std::to_string(results->total) + "], ";
      msg +=
          string("result_num [") + std::to_string(results->result_num) + "], ";
      int result_num = results->result_num > 10 ? 10 : results->result_num;
      for (int j = 0; j < result_num; ++j) {
        ResultItem *result_item = GetResultItem(results, j);
        msg += string("score [") + std::to_string(result_item->score) + "], ";
        printDoc(result_item->doc, msg);
        msg += "\n";
      }
      if (abs(GetResultItem(results, 0)->score - 1.0) < 0.001) {
        if (idx % 10000 == 0) {
          LOG(INFO) << msg;
        }
      } else {
        // if (!bitmap::test(opt.docids_bitmap_, ii)) {
        //   LOG(ERROR) << msg;
        //   error += std::to_string(ii) + ",";
        //   bitmap::set(opt.docids_bitmap_, ii);
        //   failed_count++;
        // }

        LOG(ERROR) << msg;
        error += std::to_string(ii) + ",";
        failed_count++;
      }
    }

    DestroyRequest(request);
    DestroyResponse(response);
    double elap = utils::getmillisecs() - start;
    time += elap;
    if (idx % 10000 == 0) {
      LOG(INFO) << "search time [" << time / 10000 << "]ms";
      time = 0;
    }
    idx += req_num;
  }
  LOG(ERROR) << error;
  return failed_count;
}

void *CreateEngine(string &path, int max_doc_size) {
  string log_dir = "logs";
  ByteArray *ba = StringToByteArray(log_dir);
  SetLogDictionary(ba);
  DestroyByteArray(ba);
  Config *config = MakeConfig(StringToByteArray(path), max_doc_size);
  void *engine = Init(config);
  DestroyConfig(config);
  return engine;
}

int CreateTable(void *engine, string &name, string store_type = "Mmap") {
  ByteArray *table_name = MakeByteArray(name.c_str(), name.size());
  FieldInfo **field_infos = MakeFieldInfos(opt.fields_vec.size());

  for (size_t i = 0; i < opt.fields_vec.size(); ++i) {
    BOOL do_index = TRUE;
    if (opt.fields_type[i] == STRING) do_index = FALSE;
    FieldInfo *field_info = MakeFieldInfo(StringToByteArray(opt.fields_vec[i]),
                                          opt.fields_type[i], do_index);
    SetFieldInfo(field_infos, i, field_info);
  }

  VectorInfo **vectors_info = MakeVectorInfos(1);
  VectorInfo *vector_info = MakeVectorInfo(
      StringToByteArray(opt.vector_name), FLOAT, TRUE, opt.d,
      StringToByteArray(opt.model_id), StringToByteArray(opt.retrieval_type),
      StringToByteArray(store_type), StringToByteArray(opt.store_param));
  SetVectorInfo(vectors_info, 0, vector_info);

  Table *table = MakeTable(table_name, field_infos, opt.fields_vec.size(),
                           vectors_info, 1, GetIVFPQParam());
  enum ResponseCode ret = ::CreateTable(engine, table);
  DestroyTable(table);
  return ret;
}

int MakeLastNotDone(string &path) {
  std::map<std::time_t, string> folders_map;
  std::vector<std::time_t> folders_tm;
  string dump_path = path + "/dump";
  string date_time_format = "%Y-%m-%d-%H:%M:%S";
  std::vector<string> folders = utils::ls_folder(dump_path);
  for (const string &folder_name : folders) {
    struct tm result;
    strptime(folder_name.c_str(), date_time_format.c_str(), &result);

    std::time_t t = std::mktime(&result);
    folders_tm.push_back(t);
    folders_map.insert(std::make_pair(t, folder_name));
  }

  std::sort(folders_tm.begin(), folders_tm.end());
  folders.clear();
  for (const std::time_t t : folders_tm) {
    folders.push_back(dump_path + "/" + folders_map[t]);
  }
  string folder_path = folders[folders.size() - 1];
  const string done_file = folder_path + "/dump.done";
  LOG(INFO) << "done_file=" << done_file;
  if (utils::get_file_size(done_file.c_str()) >= 0) {
    return remove(done_file.c_str());
  }
  return 0;
}

void BuildIdx(void *engine) {
  LOG(INFO) << "begin to build index";
  std::thread t(::BuildIndex, engine);
  t.detach();
  while (GetIndexStatus(engine) != INDEXED) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
}

void CreateMultiTable() {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  int max_doc_size = 10000 * 2000;
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  // Sleep(20 * 1000);

  for (int i = 0; i < 1; i++) {
    LOG(INFO) << "------------------create table-------------------id=" << i;
    void *engine = CreateEngine(root_path, max_doc_size);
    ASSERT_NE(nullptr, engine);
    ASSERT_EQ(0, CreateTable(engine, table_name));
    // Sleep(10 * 1000);
    LOG(INFO) << "------------------close--------------------id=" << i;
    EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
    ASSERT_EQ(0, Dump(engine));
    Close(engine);
    engine = nullptr;
    // Sleep(10 * 1000);
  }
  // Sleep(1000 * 1000);
}

TEST(Engine, DumpNormal) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  int max_doc_size = 10000 * 10;
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  LOG(INFO) << "------------------create table and close--------------------";
  void *engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------load no data--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));
  ASSERT_EQ(0, Load(engine));
  // BuildIdx(engine);

  LOG(INFO) << "------------------add doc and dump--------------------";
  EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------load data--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  LOG(INFO) << "------------------add_dump_add_dump-- ------------------";
  EXPECT_EQ(0, AddDoc(engine, 1 * 10000, 2 * 10000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  EXPECT_EQ(0, AddDoc(engine, 2 * 10000, 3 * 10000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------reload--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  ASSERT_EQ(0, SearchThread(engine, 3 * 10000, 0));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

void TestDumpNotDone(const string &store_type) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  int max_doc_size = 10000 * 10;
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  LOG(INFO) << "------------------init--------------------";
  void *engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  BuildIdx(engine);

  LOG(INFO) << "------------------dump and close--------------------";
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------remove dump.done file--------------------";
  ASSERT_EQ(0, MakeLastNotDone(root_path));

  // load
  LOG(INFO) << "------------------load--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  // BuildIdx(engine);

  // recreate table and add doc
  LOG(INFO) << "------------------re-create table and "
               "add_dump_add_dump--------------------";
  EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  EXPECT_EQ(0, AddDoc(engine, 1 * 10000, 2 * 10000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  // remove dump.done file
  LOG(INFO) << "------------------remove dump.done file of the last "
               "dump--------------------";
  ASSERT_EQ(0, MakeLastNotDone(root_path));

  LOG(INFO) << "------------------reload--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  // search
  ASSERT_EQ(0, SearchThread(engine, 1 * 10000, 0));
  ASSERT_EQ(3, SearchThread(engine, 3, 1 * 10000));

  // readd
  EXPECT_EQ(0, AddDoc(engine, 1 * 10000, 1 * 10000 + 3));
  Sleep(5000);

  ASSERT_EQ(0, SearchThread(engine, 1 * 10000 + 3, 0));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

TEST(Engine, DumpNotDone_Mmap) { TestDumpNotDone("Mmap"); }

TEST(Engine, DumpNotDone_RocksDB) { TestDumpNotDone("RocksDB"); }

TEST(Engine, CreateTableFromLocal) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  int max_doc_size = 10000 * 10;
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  LOG(INFO) << "------------------create table and close--------------------";
  void *engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------load no data and create table form "
               "local--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, Load(engine));

  LOG(INFO) << "------------------add doc--------------------";
  EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  BuildIdx(engine);

  LOG(INFO) << "------------------dump and close--------------------";
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------load data and create table form "
               "local--------------------";
  engine = CreateEngine(root_path, max_doc_size);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, Load(engine));

  // search
  ASSERT_EQ(0, SearchThread(engine, 1 * 10000, 0));
  ASSERT_EQ(3, SearchThread(engine, 3, 1 * 10000));

  // readd
  EXPECT_EQ(0, AddDoc(engine, 1 * 10000, 2 * 10000));
  BuildIdx(engine);

  ASSERT_EQ(0, SearchThread(engine, 2 * 10000, 0));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace Test
