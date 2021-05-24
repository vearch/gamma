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
#include "c_api/api_data/gamma_engine_status.h"
#include "c_api/api_data/gamma_request.h"
#include "c_api/api_data/gamma_response.h"
#include "c_api/api_data/gamma_table.h"
#include "gamma_api.h"
#include "test.h"

namespace Test {

using namespace std;
using namespace tig_gamma;

struct TestDumpOptions {
  TestDumpOptions() {
    nprobe = 10;
    doc_id = 0;
    d = 512;
    max_doc_size = 10000 * 10;
    search_num = 10000 * 10;
    fields_vec = {"_id", "img_url", "cid1", "cid2", "cid3"};
    fields_type = {tig_gamma::DataType::LONG, tig_gamma::DataType::STRING,
                   tig_gamma::DataType::INT, tig_gamma::DataType::INT,
                   tig_gamma::DataType::INT};
    fields_do_index = {false, false, false, false, false};
    vector_name = "abc";
    path = "files";
    model_id = "model";
    retrieval_type = "IVFPQ";  // "HNSW";  // "FLAT";
    store_type = "Mmap";
    store_param = "{\"cache_size\": 256, \"segment_size\": 1000}";
    profiles.resize(search_num * fields_vec.size());
    feature = new float[d * search_num];
    print_doc = true;
    log_dir = "test_dump_logs";
    utils::remove_dir(log_dir.c_str());
  }
  ~TestDumpOptions() {
    if (feature) {
      delete[] feature;
    }
  }

  int nprobe;
  int doc_id;
  int d;
  int max_doc_size;
  int search_num;
  std::vector<string> fields_vec;
  std::vector<tig_gamma::DataType> fields_type;
  std::vector<bool> fields_do_index;
  string path;
  string log_dir;
  string vector_name;
  string model_id;
  string retrieval_type;
  string store_type;
  string store_param;
  bool print_doc;

  std::vector<string> profiles;
  float *feature;

  char *docids_bitmap_;
};

static struct TestDumpOptions opt;

string profile_file = "./profile_10w.txt";
string feature_file = "./feat_10w.dat";

int AddDoc(void *engine, int start_id, int end_id, int interval = 0,
           long fet_offset = 0) {
  FILE *fet_fp = fopen(feature_file.c_str(), "rb");
  if (fet_fp == nullptr) {
    cerr << "open feature file error" << endl;
    return -1;
  }
  if (fet_offset == 0) {
    fet_offset = start_id * opt.d * sizeof(float);
  }
  if (fseek(fet_fp, fet_offset, SEEK_SET)) {
    cerr << "fseek error, offset=" << fet_offset << endl;
    return -1;
  }
  cerr << "add feature file offset=" << fet_offset << endl;
  std::ifstream fin;
  fin.open(profile_file.c_str());
  std::string str;
  long docid = start_id;

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
    if (i < start_id) {
      continue;
    }

    float vector[opt.d];
    size_t ret = fread((void *)vector, sizeof(float), opt.d, fet_fp);
    assert(ret == (size_t)opt.d);
    if (i < opt.search_num) {
      memcpy((void *)(opt.feature + i * opt.d), (void *)vector,
             sizeof(float) * opt.d);
    }
    tig_gamma::Doc doc;
    for (size_t j = 0; j < opt.fields_vec.size(); ++j) {
      tig_gamma::Field field;
      field.name = opt.fields_vec[j];
      field.datatype = opt.fields_type[j];

      string &data = opt.profiles[(uint64_t)i * opt.fields_vec.size() + j];
      if (opt.fields_vec[j] == "_id") {
        field.value = std::string((char *)(&docid), sizeof(long));
        docid++;
      } else if (opt.fields_type[j] == tig_gamma::DataType::INT) {
        int v = atoi(data.c_str());
        field.value = std::string((char *)(&v), sizeof(v));
      } else if (opt.fields_type[j] == tig_gamma::DataType::LONG) {
        long v = atol(data.c_str());
        field.value = std::string((char *)(&v), sizeof(v));
      } else {
        // field.value = data + "\001all";
        field.value = data;
      }

      field.source = "";
      doc.AddField(std::move(field));
    }

    tig_gamma::Field field;
    field.name = opt.vector_name;
    field.datatype = tig_gamma::DataType::VECTOR;
    field.source = "";
    int len = opt.d * sizeof(float);
    if (opt.retrieval_type == "BINARYIVF") {
      len = opt.d * sizeof(char) / 8;
    }
    field.value = std::string((char *)(vector), len);
    doc.AddField(std::move(field));

    char *doc_str = nullptr;
    int doc_len = 0;
    doc.Serialize(&doc_str, &doc_len);
    AddOrUpdateDoc(engine, doc_str, doc_len);
    free(doc_str);
    ++opt.doc_id;
    double elap = utils::getmillisecs() - start;
    if (i % 1000 == 0) {
      cerr << "AddDoc use [" << elap << "]ms" << endl;
    }
    if (interval > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
  }
  fin.close();
  fclose(fet_fp);
  return 0;
}

int DeleteDoc(void *engine, long start_id, long end_id) {
  for (long i = start_id; i < end_id; ++i) {
    int ret = ::DeleteDoc(engine, (const char *)&i, (int)sizeof(long));
    assert(ret == 0);
  }
  cerr << "delete start id=" << start_id << ", end id=" << end_id << endl;
  return 0;
}

void PrintDoc(struct tig_gamma::ResultItem &result_item, std::string &msg,
              struct TestDumpOptions &opt) {
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

int SearchThread(void *engine, int num, int start_id, long fet_offset = 0,
                 string retrieval_type="") {
  FILE *fet_fp = fopen(feature_file.c_str(), "rb");
  if (fet_fp == nullptr) {
    LOG(ERROR) << "open feature file error";
    return -1;
  }
  if (fet_offset == 0) {
    fet_offset = start_id * opt.d * sizeof(float);
  }

  if (fseek(fet_fp, fet_offset, SEEK_SET)) {
    LOG(ERROR) << "fseek error, offset=" << fet_offset;
    return -1;
  }
  LOG(INFO) << "search feature file offset=" << fet_offset;
  int idx = start_id;
  double time = 0;
  int failed_count = 0;
  int req_num = 1;
  string error;
  float *feature = new float[opt.d * req_num];
  int end_id = start_id + num;
  while (idx < end_id) {
    double start = utils::getmillisecs();
    struct tig_gamma::VectorQuery vector_query;
    vector_query.name = opt.vector_name;

    int len = opt.d * sizeof(float) * req_num;
    if (opt.retrieval_type == "BINARYIVF") {
      len = opt.d * sizeof(char) / 8 * req_num;
    }
    int ret =
        (int)fread((void *)feature, sizeof(float) * opt.d, req_num, fet_fp);
    assert(ret == req_num);
    char *value = reinterpret_cast<char *>(feature);
    vector_query.value = std::string(value, len);

    vector_query.min_score = 0;
    vector_query.max_score = 10000;
    vector_query.boost = 0.1;
    vector_query.has_boost = 0;
    vector_query.retrieval_type = retrieval_type;

    tig_gamma::Request request;
    request.SetTopN(10);
    request.AddVectorQuery(vector_query);
    request.SetReqNum(req_num);
    request.SetBruteForceSearch(0);
    request.SetHasRank(true);
    std::string retrieval_params =
        "{\"metric_type\" : \"InnerProduct\", \"recall_num\" : "
        "10, \"nprobe\" : 10, \"ivf_flat\" : 0, \"efSearch\": 100}";
    request.SetRetrievalParams(retrieval_params);
    // request.SetOnlineLogLevel("");
    request.SetMultiVectorRank(0);
    request.SetL2Sqrt(false);

    char *request_str, *response_str;
    int request_len, response_len;

    request.Serialize(&request_str, &request_len);
    ret =
        Search(engine, request_str, request_len, &response_str, &response_len);

    assert(ret == 0);
    free(request_str);

    tig_gamma::Response response;
    response.Deserialize(response_str, response_len);

    free(response_str);

    if (opt.print_doc) {
      std::vector<struct tig_gamma::SearchResult> &results = response.Results();
      assert(results.size() > 0);
      for (size_t i = 0; i < results.size(); ++i) {
        int ii = idx + i;
        string msg = std::to_string(ii) + ", ";
        struct tig_gamma::SearchResult &result = results[i];

        std::vector<struct tig_gamma::ResultItem> &result_items =
            result.result_items;
        assert(result_items.size() > 0);
        msg += string("total [") + std::to_string(result.total) + "], ";
        msg += string("result_num [") + std::to_string(result_items.size()) +
               "], ";
        for (size_t j = 0; j < result_items.size(); ++j) {
          struct tig_gamma::ResultItem &result_item = result_items[j];
          PrintDoc(result_item, msg, opt);
          msg += "\n";
        }
        if (abs(result_items[0].score - 1.0) < 0.001) {
          if (ii % 1000 == 0) {
            LOG(INFO) << msg << endl;
          }
        } else {
          LOG(ERROR) << msg;
          error += std::to_string(ii) + ",";
          failed_count++;
        }
      }
    }
    double elap = utils::getmillisecs() - start;
    time += elap;
    if (idx % 10000 == 0) {
      LOG(INFO) << "search time [" << time / 10000 << "]ms";
      time = 0;
    }
    idx += req_num;
  }
  delete[] feature;
  LOG(ERROR) << error;
  return failed_count;
}

void *CreateEngine(string &path) {
  tig_gamma::Config config;
  config.SetPath(path);
  config.SetLogDir(opt.log_dir);

  char *config_str = nullptr;
  int len = 0;
  config.Serialize(&config_str, &len);
  void *engine = Init(config_str, len);
  free(config_str);
  return engine;
}

int CreateTable(void *engine, string &name, string store_type = "MemoryOnly",
                bool multi_model = false) {
  tig_gamma::TableInfo table;
  table.SetName(name);
  table.SetIndexingSize(10000);
  if (multi_model) {
    vector<string> retrieval_types = {"IVFPQ", "HNSW"};
    vector<string> retrieval_params = {kIVFPQParam, kHNSWParam_str};
    table.SetRetrievalTypes(retrieval_types);
    table.SetRetrievalParams(retrieval_params);
  } else {
    table.SetRetrievalType(opt.retrieval_type);
    table.SetRetrievalParam(kIVFPQParam);
  }

  for (size_t i = 0; i < opt.fields_vec.size(); ++i) {
    struct tig_gamma::FieldInfo field_info;
    field_info.name = opt.fields_vec[i];

    field_info.is_index = opt.fields_do_index[i];
    field_info.data_type = opt.fields_type[i];
    table.AddField(field_info);
  }

  struct tig_gamma::VectorInfo vector_info;
  vector_info.name = opt.vector_name;
  vector_info.data_type = tig_gamma::DataType::FLOAT;
  vector_info.is_index = true;
  vector_info.dimension = opt.d;
  vector_info.model_id = "";
  vector_info.store_type = store_type;
  vector_info.store_param = "{\"cache_size\": 2048}";
  vector_info.has_source = false;

  table.AddVectorInfo(vector_info);

  char *table_str = nullptr;
  int len = 0;
  table.Serialize(&table_str, &len);

  int ret = ::CreateTable(engine, table_str, len);

  free(table_str);

  return ret;
}

int MakeLastNotDone(string &path) {
  std::map<std::time_t, string> folders_map;
  std::vector<std::time_t> folders_tm;
  string dump_path = path + "/retrieval_model_index";
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
  ::BuildIndex(engine);
  int n_index_status = -1;
  do {
    char *status = nullptr;
    int len = 0;
    GetEngineStatus(engine, &status, &len);
    tig_gamma::EngineStatus engine_status;
    engine_status.Deserialize(status, len);
    free(status);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    n_index_status = engine_status.IndexStatus();
  } while (n_index_status != 2);
}

void CreateMultiTable() {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  // int max_doc_size = 10000 * 2000;
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  // Sleep(20 * 1000);

  for (int i = 0; i < 1; i++) {
    LOG(INFO) << "------------------create table-------------------id=" << i;
    void *engine = CreateEngine(root_path);
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

void TestDumpNormal(const string &store_type) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  cout << "------------------create table and close--------------------"
       << endl;
  void *engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  Close(engine);
  engine = nullptr;

  cout << "------------------load no data--------------------" << endl;
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));

  cout << "------------------add doc and dump--------------------" << endl;
  EXPECT_EQ(0, AddDoc(engine, 0, 10000));
  BuildIdx(engine);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  cout << "------------------load data--------------------" << endl;
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  LOG(INFO) << "------------------add_dump_add_dump-- ------------------";
  EXPECT_EQ(0, AddDoc(engine, 10000, 11000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  EXPECT_EQ(0, AddDoc(engine, 11000, 12000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------reload--------------------";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  ASSERT_EQ(0, SearchThread(engine, 12000, 0));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

TEST(Engine, DumpNormal_MemoryOnly) { TestDumpNormal("MemoryOnly"); }

TEST(Engine, DumpNormal_Mmap) { TestDumpNormal("Mmap"); }

TEST(Engine, DumpNormal_RocksDB) { TestDumpNormal("RocksDB"); }

void TestDumpNotDone(const string &store_type) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  cerr << "------------------init--------------------";
  void *engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  EXPECT_EQ(0, AddDoc(engine, 0, 10000));
  BuildIdx(engine);

  cerr << "------------------dump and close--------------------";
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  cerr << "------------------remove dump.done file--------------------";
  ASSERT_EQ(0, MakeLastNotDone(root_path));

  cerr << "------------------load--------------------";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  cerr << "------------------add_dump and close--------------------";
  EXPECT_EQ(0, AddDoc(engine, 10000, 11000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  cerr << "------------------reload--------------------";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  cerr << "------------------search--------------------";
  ASSERT_EQ(0, SearchThread(engine, 11000, 0));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

TEST(Engine, DumpNotDone_MemoryOnly) { TestDumpNotDone("MemoryOnly"); }

TEST(Engine, DumpNotDone_Mmap) { TestDumpNotDone("Mmap"); }

TEST(Engine, DumpNotDone_RocksDB) { TestDumpNotDone("RocksDB"); }

TEST(Engine, CreateTableFromLocal) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  cerr << "------------------create table--------------------\n";
  void *engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name));

  cerr << "------------------add doc--------------------\n";
  EXPECT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  BuildIdx(engine);

  cerr << "------------------dump and close--------------------\n";
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  cerr << "------------------load data and create table from "
          "local--------------------\n";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, Load(engine));

  cerr << "------------------readd doc--------------------\n";
  EXPECT_EQ(0, AddDoc(engine, 1 * 10000, 11000));
  BuildIdx(engine);

  cerr << "------------------search--------------------\n";
  ASSERT_EQ(0, SearchThread(engine, 11000, 0));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

TEST(Engine, UpdateAndCompactIndex) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_compact_index";
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;

  LOG(INFO) << "------------------create table--------------------";
  void *engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, "MemoryOnly"));

  LOG(INFO) << "------------------add doc and build--------------------";
  ASSERT_EQ(0, AddDoc(engine, 0, 1 * 10000));
  BuildIdx(engine);

  LOG(INFO) << "------------------update docs--------------------";
  long fet_offset = (long)50000 * opt.d * 4;
  ASSERT_EQ(0, AddDoc(engine, 3000, 5000, 0, fet_offset));

  Sleep(1000 * 10);
  ASSERT_EQ(0, SearchThread(engine, 2000, 3000, fet_offset));

  LOG(INFO) << "------------------delete docs--------------------";
  ASSERT_EQ(0, DeleteDoc(engine, 0, 4000));

  Sleep(1000 * 6);
  LOG(INFO) << "------------------add docs--------------------";
  ASSERT_EQ(0, AddDoc(engine, 10000, 14000));

  LOG(INFO) << "------------------dump and close--------------------";
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------load--------------------";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, "MemoryOnly"));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  LOG(INFO) << "------------------final search--------------------";
  Sleep(1000 * 5);
  fet_offset += (long)1000 * opt.d * sizeof(float);
  ASSERT_EQ(0, SearchThread(engine, 1000, 4000, fet_offset));
  ASSERT_EQ(0, SearchThread(engine, 9000, 5000));

  Close(engine);
  engine = nullptr;
}

TEST(Engine, MultiModel) {
  string case_name = GetCurrentCaseName();
  string table_name = "test_table";
  utils::remove_dir(case_name.c_str());
  utils::make_dir(case_name.c_str());
  string root_path = "./" + case_name;
  string store_type = "MemoryOnly";

  cout << "------------------create table and close--------------------"
       << endl;
  void *engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type, true));
  // Close(engine);
  // engine = nullptr;

  cout << "------------------add doc and dump--------------------" << endl;
  EXPECT_EQ(0, AddDoc(engine, 0, 10000));
  BuildIdx(engine);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  cout << "------------------load data--------------------" << endl;
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type, true));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  LOG(INFO) << "------------------add_dump_add_dump-- ------------------";
  EXPECT_EQ(0, AddDoc(engine, 10000, 11000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  EXPECT_EQ(0, AddDoc(engine, 11000, 12000));
  Sleep(1000);
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;

  LOG(INFO) << "------------------reload--------------------";
  engine = CreateEngine(root_path);
  ASSERT_NE(nullptr, engine);
  ASSERT_EQ(0, CreateTable(engine, table_name, store_type, true));
  ASSERT_EQ(0, Load(engine));
  BuildIdx(engine);

  RandomGenerator rg;
  string retrieval_type = rg.Rand(2) == 0 ? "IVFPQ" : "HNSW";
  ASSERT_EQ(0, SearchThread(engine, 12000, 0, 0, retrieval_type));
  ASSERT_EQ(0, Dump(engine));
  Close(engine);
  engine = nullptr;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace Test
