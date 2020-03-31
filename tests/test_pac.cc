/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <vector>
#include "test.h"
#include "util/utils.h"

using std::map;
using std::string;
using std::vector;

constexpr int kMaxDocSize = 1000000;

namespace tig_gamma {

struct TableInfo {
  string table_name;
  map<string, DataType> field_mappings;

  struct MetaData {
    string vector_name;
    int dimension;
    string model_id;
    string retrieval_type;
    string store_type;
  } meta;
};

class PacTest {
 public:
  explicit PacTest(int max_doc_size) : engine_(nullptr), exit_(false) {}

  ~PacTest() {
    exit_ = true;
    if (add_task_.joinable()) {
      add_task_.join();
    }

    Dump(engine_);
    Close(engine_);
  }

 public:
  void Init(const string &path, const string &log_dir) {
    engine_ = _InitEngine(path, log_dir);
  }

  void CreateTable(TableInfo t) {
    table_info_ = t;
    _CreateTable(engine_, t);
  }

  void BuildIndex() {
    std::thread t(::BuildIndex, engine_);
    t.detach();

    while (GetIndexStatus(engine_) != INDEXED) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    cerr << ">>> Build Index Done.\n";
  }

  void StartAddThread() {
    add_task_ = std::thread([this]() {
      int times = kMaxDocSize;
      while (not exit_ && times--) {
        int branch = 7;
        int product_code = 0;
        int type = 1;

        switch (times % 4) {
          case 0:
            product_code = 100;
            break;
          case 1:
            product_code = 101;
            break;
          case 2:
            product_code = 102;
            break;
          case 3:
            product_code = 102;
            type = 0;
            break;
        }

        _AddDoc(engine_, table_info_,
                {
                    {"branch", branch},
                    {"product_code", product_code},
                    {"type", type},
                });
      }
      cerr << ">>> Add finished.\n";
    });
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  void Search(int times) {
    while (times--) {
      _Search(engine_, table_info_);
    }
  }

  void GetDoc(int doc_id) {
    ByteArray *key = StringToByteArray(std::to_string(doc_id));
    Doc *doc = GetDocByID(engine_, key);
    string msg;
    printDoc(doc, msg);
    cerr << msg << "\n";
  }

 private:
  static void *_InitEngine(const string &path, const string &log_dir) {
    Config *config = MakeConfig(StringToByteArray(path), kMaxDocSize);
    SetLogDictionary(StringToByteArray(log_dir));
    void *engine = ::Init(config);
    DestroyConfig(config);
    return engine;
  }

  static void _CreateTable(void *engine, TableInfo &t) {
    int d = t.meta.dimension;
    ByteArray *table_name = StringToByteArray(t.table_name);

    FieldInfo **field_infos = MakeFieldInfos(t.field_mappings.size());
    int i = 0;
    for (auto _ : t.field_mappings) {
      ByteArray *name = StringToByteArray(_.first);
      DataType type = _.second;

      FieldInfo *field_info = MakeFieldInfo(name, type, 1);
      SetFieldInfo(field_infos, i++, field_info);
    }

    VectorInfo **vector_infos = MakeVectorInfos(1);
    VectorInfo *vector_info =
        MakeVectorInfo(StringToByteArray(t.meta.vector_name),
                       FLOAT,  // data_type
                       TRUE,
                       d,  // dimension
                       StringToByteArray(t.meta.model_id),
                       StringToByteArray(t.meta.retrieval_type),
                       StringToByteArray(t.meta.store_type), nullptr);
    SetVectorInfo(vector_infos, 0, vector_info);

    Table *table = MakeTable(table_name,
                             field_infos,              // fields
                             t.field_mappings.size(),  // fields_num
                             vector_infos,             // vectors_info
                             1, kIVFPQParam);          // vectors_num
    ResponseCode code = ::CreateTable(engine, table);
    DestroyTable(table);

    if (code != 0) {
      printf("create table ret [%d]\n", code);
    }
  }

  static void _AddDoc(void *engine, TableInfo &t, map<string, int> vm) {
    static int doc_id = 0;
    int field_num = t.field_mappings.size() + 1;  // the last one is VECTOR

    int d = t.meta.dimension;
    Field **fields = MakeFields(field_num);
    int i = 0;
    for (auto _ : t.field_mappings) {
      ByteArray *name = StringToByteArray(_.first);
      DataType type = _.second;

      ByteArray *value = nullptr;
      if (_.first == "_id") {
        value = StringToByteArray("jsf/img/" + std::to_string(doc_id));
        doc_id++;  // increment
      } else {
        if (type == INT) {
          value = ToByteArray<int>(vm[_.first]);
        } else {
          cerr << "not support.\n";
        }
      }

      Field *field = MakeField(name, value, NULL, type);
      SetField(fields, i++, field);
    }

    vector<float> xb(d * 1);
    srand48(doc_id);
    std::generate(xb.begin(), xb.end(), drand48);
    ByteArray *value = FloatToByteArray(xb.data(), d);

    Field *field =
        MakeField(StringToByteArray(t.meta.vector_name), value, NULL, VECTOR);
    SetField(fields, i++, field);

    Doc *doc = MakeDoc(fields, field_num);
    ResponseCode code = AddDoc(engine, doc);
    DestroyDoc(doc);

    if (code != 0) {
      printf("add doc ret [%d]\n", code);
    }
  }

  static void _Search(void *engine, TableInfo &t) {
    int d = t.meta.dimension;
    VectorQuery **querys = MakeVectorQuerys(1);

    vector<float> xb(d * 1);
    srand48(d);
    std::generate(xb.begin(), xb.end(), drand48);

    VectorQuery *query =
        MakeVectorQuery(StringToByteArray(t.meta.vector_name),  // name
                        FloatToByteArray(xb.data(), d),         // value
                        0,                                      // min_score
                        10000,                                  // max_score
                        1, 0);                                  // boost
    SetVectorQuery(querys, 0, query);

    RangeFilter **filters = MakeRangeFilters(3);

    auto add_filter = [filters](int index, const string &field, int l,
                                int u = -1) -> void {
      u = (u < 0) ? l : u;
      RangeFilter *filter =
          MakeRangeFilter(StringToByteArray(field),              // field
                          StringToByteArray(std::to_string(l)),  // lower_value
                          StringToByteArray(std::to_string(u)),  // upper_value
                          1,   // include_lower
                          1);  // include_upper
      SetRangeFilter(filters, index, filter);
    };

    add_filter(0, "branch", 7);
    add_filter(1, "product_code", 101);
    add_filter(2, "type", 1);

    Request *request = MakeRequest(10,      // topk
                                   querys,  // vector querys
                                   1,       // vector querys num
                                   nullptr, // fields
                                   0,       // fields_num
                                   filters, // range_filters
                                   3,       // range_filters_num
                                   nullptr, // term_filters
                                   0,       // term_filters_num
                                   1,       // req_num
                                   0,       // direct_search_threashold
                                   nullptr, TRUE, 0);

    Response *response = ::Search(engine, request);

    for (int i = 0; i < response->req_num; i++) {
      SearchResult *result = GetSearchResult(response, i);
      printf("request [%d] total call %d\n", i, result->total);

      for (int j = 0; j < result->result_num; j++) {
        ResultItem *result_item = GetResultItem(result, j);
        string msg =
            string("score [") + std::to_string(result_item->score) + "], ";
        printDoc(result_item->doc, msg);
        printf("%s\n", msg.c_str());
      }

      printf("response %s\n",
             string(result->msg->value, result->msg->len).c_str());
    }

    DestroyResponse(response);
  }

 private:
  TableInfo table_info_;
  void *engine_;

  std::thread add_task_;
  volatile bool exit_;
};

}  // namespace tig_gamma

using tig_gamma::PacTest;

void test_pac() {
  tig_gamma::TableInfo t;
  t.table_name = "pac";
  t.field_mappings = {
      {"_id", STRING},
      {"branch", INT},
      {"product_code", INT},
      {"type", INT},
  };

  t.meta.vector_name = "pac";
  t.meta.dimension = 5000;
  t.meta.retrieval_type = "PACINS";
  t.meta.store_type = "MemoryOnly";
  t.meta.model_id = "model";

  PacTest pt(kMaxDocSize);

  pt.Init("table", "logs");
  pt.CreateTable(t);
  cerr << "Init & CreateTable done!\n";

  pt.StartAddThread();
  pt.BuildIndex();

  // pt.GetDoc(1);
  pt.Search(1);
}

int main(int argc, char *argv[]) {
  test_pac();
  return 0;
}
