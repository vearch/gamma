/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_engine.h"

#include <fcntl.h>
#include <locale.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <string.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <vector>

#include "bitmap.h"
#include "cJSON.h"
#include "error_code.h"
#include "gamma_common_data.h"
#include "gamma_table_io.h"
#include "log.h"
#include "raw_vector_io.h"
#include "table_io.h"
#include "utils.h"

using std::string;
using namespace tig_gamma::table;

namespace tig_gamma {

#ifdef DEBUG
static string float_array_to_string(float *data, int len) {
  if (data == nullptr) return "";
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < len; ++i) {
    ss << data[i];
    if (i != len - 1) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

static string VectorQueryToString(VectorQuery *vector_query) {
  std::stringstream ss;
  ss << "name:"
     << std::string(vector_query->name->value, vector_query->name->len)
     << " min score:" << vector_query->min_score
     << " max score:" << vector_query->max_score
     << " boost:" << vector_query->boost
     << " has boost:" << vector_query->has_boost << " value:"
     << float_array_to_string((float *)vector_query->value->value,
                              vector_query->value->len / sizeof(float));
  return ss.str();
}

// static string RequestToString(const Request *request) {
//   std::stringstream ss;
//   ss << "{req_num:" << request->req_num << " topn:" << request->topn
//      << " has_rank:" << request->has_rank
//      << " vec_num:" << request->vec_fields_num;
//   for (int i = 0; i < request->vec_fields_num; ++i) {
//     ss << " vec_id:" << i << " [" <<
//     VectorQueryToString(request->vec_fields[i])
//        << "]";
//   }
//   ss << "}";
//   return ss.str();
// }
#endif  // DEBUG

#ifndef __APPLE__
static std::thread *gMemTrimThread = nullptr;
void MemTrimHandler() {
  LOG(INFO) << "memory trim thread start......";
  while (1) {
    malloc_trim(0);
    std::this_thread::sleep_for(std::chrono::seconds(60));  // 1 minute
  }
  LOG(INFO) << "memory trim thread exit!";
}
#endif

GammaEngine::GammaEngine(const string &index_root_path)
    : index_root_path_(index_root_path),
      date_time_format_("%Y-%m-%d-%H:%M:%S") {
  docids_bitmap_ = nullptr;
  table_ = nullptr;
  vec_manager_ = nullptr;
  index_status_ = IndexStatus::UNINDEXED;
  // max_doc_size_ = init_max_doc_size;
  delete_num_ = 0;
  b_running_ = false;
  b_field_running_ = false;
  dump_docid_ = 0;
  bitmap_bytes_size_ = 0;
  field_range_index_ = nullptr;
  created_table_ = false;
  indexed_field_num_ = 0;
  b_loading_ = false;
#ifdef PERFORMANCE_TESTING
  search_num_ = 0;
#endif
  af_exector_ = nullptr;
}

GammaEngine::~GammaEngine() {
  if (b_running_) {
    b_running_ = false;
    std::mutex running_mutex;
    std::unique_lock<std::mutex> lk(running_mutex);
    running_cv_.wait(lk);
  }

  if (b_field_running_) {
    b_field_running_ = false;
    std::mutex running_mutex;
    std::unique_lock<std::mutex> lk(running_mutex);
    running_field_cv_.wait(lk);
  }

  if (af_exector_) {
    af_exector_->Stop();
    CHECK_DELETE(af_exector_);
  }
  CHECK_DELETE(table_io_);

  if (vec_manager_) {
    delete vec_manager_;
    vec_manager_ = nullptr;
  }

  if (table_) {
    delete table_;
    table_ = nullptr;
  }

  if (docids_bitmap_) {
    delete docids_bitmap_;
    docids_bitmap_ = nullptr;
  }

  if (field_range_index_) {
    delete field_range_index_;
    field_range_index_ = nullptr;
  }
}

GammaEngine *GammaEngine::GetInstance(const string &index_root_path) {
  GammaEngine *engine = new GammaEngine(index_root_path);
  int ret = engine->Setup();
  if (ret < 0) {
    LOG(ERROR) << "BuildSearchEngine [" << index_root_path << "] error!";
    return nullptr;
  }
  return engine;
}

int GammaEngine::Setup() {
  if (!utils::isFolderExist(index_root_path_.c_str())) {
    mkdir(index_root_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  dump_path_ = index_root_path_ + "/retrieval_model_index";
  if (!utils::isFolderExist(dump_path_.c_str())) {
    mkdir(dump_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  if (!docids_bitmap_) {
    if (bitmap::create(docids_bitmap_, bitmap_bytes_size_,
                       std::numeric_limits<int>::max()) != 0) {
      LOG(ERROR) << "Cannot create bitmap!";
      return INTERNAL_ERR;
    }
  }

  if (!table_) {
    table_ = new Table(index_root_path_);
  }

  if (!vec_manager_) {
    vec_manager_ = new VectorManager(VectorStorageType::Mmap, docids_bitmap_,
                                     index_root_path_);
  }

#ifndef __APPLE__
  if (gMemTrimThread == nullptr) {
    gMemTrimThread = new std::thread(MemTrimHandler);
    if (gMemTrimThread) {
      gMemTrimThread->detach();
    } else {
      LOG(ERROR) << "create memory trim thread error";
    }
  }
#endif

  max_docid_ = 0;
  LOG(INFO) << "GammaEngine setup successed! bitmap_bytes_size="
            << bitmap_bytes_size_;
  return 0;
}

int GammaEngine::Search(Request &request, Response &response_results) {
#ifdef DEBUG
// LOG(INFO) << "search request:" << RequestToString(request);
#endif

  int ret = 0;
  int req_num = request.ReqNum();

  if (req_num <= 0) {
    string msg = "req_num should not less than 0";
    LOG(ERROR) << msg;
    return -1;
  }

  // TODO: it may be opened later
  // utils::OnlineLogger logger;
  // if (0 != logger.Init(online_log_level)) {
  //   LOG(WARNING) << "init online logger error!";
  // }

  int topn = request.TopN();
  bool brute_force_search = ((request.BruteForceSearch() == 1) ||
                             ((request.BruteForceSearch() == 0) &&
                              (index_status_ != IndexStatus::INDEXED)));

  if ((not brute_force_search) && (index_status_ != IndexStatus::INDEXED)) {
    string msg = "index not trained!";
    LOG(ERROR) << msg;
    for (int i = 0; i < req_num; ++i) {
      SearchResult result;
      result.msg = msg;
      result.result_code = SearchResultCode::INDEX_NOT_TRAINED;
      response_results.AddResults(std::move(result));
    }
    return -2;
  }

  std::vector<struct VectorQuery> &vec_fields = request.VecFields();
  GammaQuery gamma_query;
  gamma_query.vec_query = vec_fields;

  gamma_query.condition = new GammaSearchCondition;
  gamma_query.condition->topn = topn;
  gamma_query.condition->multi_vector_rank =
      request.MultiVectorRank() == 1 ? true : false;
  gamma_query.condition->brute_force_search = brute_force_search;
  gamma_query.condition->l2_sqrt = request.L2Sqrt();
  gamma_query.condition->retrieval_parameters = request.RetrievalParams();
  gamma_query.condition->has_rank = request.HasRank();

#ifdef BUILD_GPU
  gamma_query.condition->range_filters = request.RangeFilters();
  gamma_query.condition->term_filters = request.TermFilters();
  gamma_query.condition->table = table_;
#endif  // BUILD_GPU

#ifndef BUILD_GPU
  MultiRangeQueryResults range_query_result;
  std::vector<struct RangeFilter> &range_filters = request.RangeFilters();
  size_t range_filters_num = range_filters.size();

  std::vector<struct TermFilter> &term_filters = request.TermFilters();
  size_t term_filters_num = term_filters.size();
  if (range_filters_num > 0 || term_filters_num > 0) {
    int num = MultiRangeQuery(request, gamma_query.condition, response_results,
                              &range_query_result);
    if (num == 0) {
      return 0;
    }
  }
#ifdef PERFORMANCE_TESTING
  gamma_query.condition->GetPerfTool().Perf("filter");
#endif
#endif

  size_t vec_fields_num = vec_fields.size();
  if (vec_fields_num > 0) {
    GammaResult gamma_results[req_num];
    int doc_num = GetDocsNum();

    for (int i = 0; i < req_num; ++i) {
      gamma_results[i].total = doc_num;
    }

    ret = vec_manager_->Search(gamma_query, gamma_results);
    if (ret != 0) {
      string msg = "search error [" + std::to_string(ret) + "]";
      for (int i = 0; i < req_num; ++i) {
        SearchResult result;
        result.msg = msg;
        result.result_code = SearchResultCode::SEARCH_ERROR;
        response_results.AddResults(std::move(result));
      }
      return -3;
    }

#ifdef PERFORMANCE_TESTING
    gamma_query.condition->GetPerfTool().Perf("search total");
#endif
    PackResults(gamma_results, response_results, request);
#ifdef PERFORMANCE_TESTING
    gamma_query.condition->GetPerfTool().Perf("pack results");
#endif

#ifdef BUILD_GPU
  }
#else
  } else {
    GammaResult gamma_result;
    gamma_result.topn = topn;

    std::vector<std::pair<string, int>> fields_ids;
    std::vector<string> vec_names;

    const auto range_result = range_query_result.GetAllResult();
    if (range_result == nullptr && term_filters_num > 0) {
      for (size_t i = 0; i < term_filters_num; ++i) {
        struct TermFilter &term_filter = term_filters[i];

        string value = term_filter.field;

        int doc_id = -1;
        if (table_->GetDocIDByKey(term_filter.value, doc_id) != 0) {
          continue;
        }

        fields_ids.emplace_back(std::make_pair(value, doc_id));
        vec_names.emplace_back(std::move(value));
      }
      if (fields_ids.size() > 0) {
        gamma_result.init(topn, vec_names.data(), fields_ids.size());
        std::vector<string> vec;
        int ret = vec_manager_->GetVector(fields_ids, vec);
        if (ret == 0) {
          int idx = 0;
          VectorDoc *doc = gamma_result.docs[gamma_result.results_count];
          for (const auto &field_id : fields_ids) {
            int id = field_id.second;
            doc->docid = id;
            doc->fields[idx].name = vec[idx];
            doc->fields[idx].source = nullptr;
            doc->fields[idx].source_len = 0;
            ++idx;
          }
          ++gamma_result.results_count;
          gamma_result.total = 1;
        }
      }
    } else {
      gamma_result.init(topn, nullptr, 0);
      for (int docid = 0; docid < max_docid_; ++docid) {
        if (range_query_result.Has(docid) &&
            !bitmap::test(docids_bitmap_, docid)) {
          ++gamma_result.total;
          if (gamma_result.results_count < topn) {
            gamma_result.docs[gamma_result.results_count++]->docid = docid;
          }
        }
      }
    }
    // response_results.req_num = 1;  // only one result
    PackResults(&gamma_result, response_results, request);
  }
#endif

#ifdef PERFORMANCE_TESTING
  LOG(INFO) << gamma_query.condition->GetPerfTool().OutputPerf().str();
#endif

  std::string online_log_level = request.OnlineLogLevel();
  if (strncasecmp("debug", online_log_level.c_str(), 5) == 0) {
    response_results.SetOnlineLogMessage(
        gamma_query.condition->GetPerfTool().OutputPerf().str());
  }

  return ret;
}

int GammaEngine::MultiRangeQuery(Request &request,
                                 GammaSearchCondition *condition,
                                 Response &response_results,
                                 MultiRangeQueryResults *range_query_result) {
  std::vector<FilterInfo> filters;
  std::vector<struct RangeFilter> &range_filters = request.RangeFilters();
  std::vector<struct TermFilter> &term_filters = request.TermFilters();

  int range_filters_size = range_filters.size();
  int term_filters_size = term_filters.size();

  filters.resize(range_filters_size + term_filters_size);
  int idx = 0;

  for (int i = 0; i < range_filters_size; ++i) {
    struct RangeFilter &filter = range_filters[i];

    filters[idx].field = table_->GetAttrIdx(filter.field);
    filters[idx].lower_value = filter.lower_value;
    filters[idx].upper_value = filter.upper_value;

    ++idx;
  }

  for (int i = 0; i < term_filters_size; ++i) {
    struct TermFilter &filter = term_filters[i];

    filters[idx].field = table_->GetAttrIdx(filter.field);
    filters[idx].lower_value = filter.value;
    filters[idx].is_union = static_cast<FilterOperator>(filter.is_union);

    ++idx;
  }

  int retval = field_range_index_->Search(filters, range_query_result);

  if (retval == 0) {
    string msg = "No result: numeric filter return 0 result";
    LOG(INFO) << msg;
    for (int i = 0; i < request.ReqNum(); ++i) {
      SearchResult result;
      result.msg = msg;
      result.result_code = SearchResultCode::SUCCESS;
      response_results.AddResults(std::move(result));
    }
  } else if (retval < 0) {
    condition->range_query_result = nullptr;
  } else {
    condition->range_query_result = range_query_result;
  }
  return retval;
}

int GammaEngine::CreateTable(TableInfo &table) {
  if (!vec_manager_ || !table_) {
    LOG(ERROR) << "vector and table should not be null!";
    return -1;
  }

  string dump_meta_path = index_root_path_ + "/dump.meta";
  utils::JsonParser *meta_jp = nullptr;
  utils::ScopeDeleter1<utils::JsonParser> del1;
  if (utils::file_exist(dump_meta_path)) {
    long len = utils::get_file_size(dump_meta_path);
    if (len > 0) {
      utils::FileIO fio(dump_meta_path);
      if (fio.Open("r")) {
        LOG(ERROR) << "open file error, path=" << dump_meta_path;
        return IO_ERR;
      }
      char *buf = new char[len + 1];
      buf[len] = '\0';
      if (len != fio.Read(buf, 1, len)) {
        LOG(ERROR) << "read file error, path=" << dump_meta_path;
        return IO_ERR;
      }
      meta_jp = new utils::JsonParser();
      del1.set(meta_jp);
      if (meta_jp->Parse(buf)) {
        return FORMAT_ERR;
      }
    }
  }

  int ret_vec = vec_manager_->CreateVectorTable(table, meta_jp);
  TableParams disk_table_params;
  if (meta_jp) {
    utils::JsonParser table_jp;
    meta_jp->GetObject("table", table_jp);
    disk_table_params.Parse(table_jp);
  }
  int ret_table = table_->CreateTable(table, disk_table_params);
  indexing_size_ = table.IndexingSize();
  if (ret_vec != 0 || ret_table != 0) {
    LOG(ERROR) << "Cannot create table!";
    return -2;
  }

  af_exector_ = new AsyncFlushExecutor();
  table_io_ = new TableIO(table_);
  int ret = table_io_->Init();
  if (ret) {
    return ret;
  }
  af_exector_->Add(static_cast<AsyncFlusher *>(table_io_));

  if (!meta_jp) {
    utils::JsonParser dump_meta_;
    dump_meta_.PutInt("version", 320);  // version=3.2.0

    utils::JsonParser table_jp;
    table_->GetDumpConfig()->ToJson(table_jp);
    dump_meta_.PutObject("table", std::move(table_jp));

    utils::JsonParser vectors_jp;
    for (auto &it : vec_manager_->RawVectors()) {
      DumpConfig *dc = it.second->GetDumpConfig();
      if (dc) {
        utils::JsonParser jp;
        dc->ToJson(jp);
        vectors_jp.PutObject(dc->name, std::move(jp));
      }
    }
    dump_meta_.PutObject("vectors", std::move(vectors_jp));

    utils::FileIO fio(dump_meta_path);
    fio.Open("w");
    string meta_str = dump_meta_.ToStr(true);
    fio.Write(meta_str.c_str(), 1, meta_str.size());
  }
  for (auto &it : vec_manager_->RawVectors()) {
    RawVectorIO *rio = it.second->GetIO();
    if (rio == nullptr) continue;
    AsyncFlusher *flusher = dynamic_cast<AsyncFlusher *>(rio);
    if (flusher) {
      af_exector_->Add(flusher);
    }
  }

#ifndef BUILD_GPU
  field_range_index_ = new MultiFieldsRangeIndex(index_root_path_, table_);
  if ((nullptr == field_range_index_) || (AddNumIndexFields() < 0)) {
    LOG(ERROR) << "add numeric index fields error!";
    return -3;
  }

  auto func_build_field_index = std::bind(&GammaEngine::BuildFieldIndex, this);
  std::thread t(func_build_field_index);
  t.detach();
#endif
  std::string table_name = table.Name();
  std::string path = index_root_path_ + "/" + table_name + ".schema";
  TableSchemaIO tio(path);  // rewrite it if the path is already existed
  if (tio.Write(table)) {
    LOG(ERROR) << "write table schema error, path=" << path;
  }

  af_exector_->Start();

  LOG(INFO) << "create table [" << table_name << "] success!";
  created_table_ = true;
  return 0;
}

int GammaEngine::AddOrUpdate(Doc &doc) {
#ifdef PERFORMANCE_TESTING
  double start = utils::getmillisecs();
#endif
  std::vector<struct Field> &fields_table = doc.TableFields();
  std::vector<struct Field> &fields_vec = doc.VectorFields();
  std::string &key = doc.Key();

  // add fields into table
  int docid = -1;
  table_->GetDocIDByKey(key, docid);
  if (docid == -1) {
    int ret = table_->Add(key, fields_table, max_docid_);
    if (ret != 0) return -2;
#ifndef BUILD_GPU
    for (size_t i = 0; i < fields_table.size(); ++i) {
      struct Field &field = fields_table[i];
      int idx = table_->GetAttrIdx(field.name);
      field_range_index_->Add(max_docid_, idx);
    }
#endif  // BUILD_GPU
  } else {
    if (Update(docid, fields_table, fields_vec)) {
      LOG(ERROR) << "update error, key=" << key << ", docid=" << docid;
      return -3;
    }
    return 0;
  }
#ifdef PERFORMANCE_TESTING
  double end_table = utils::getmillisecs();
#endif

  // add vectors by VectorManager
  if (vec_manager_->AddToStore(max_docid_, fields_vec) != 0) {
    return -4;
  }
  if (not b_running_ and index_status_ == UNINDEXED) {
    if (max_docid_ >= indexing_size_) {
      LOG(INFO) << "Begin indexing.";
      this->BuildIndex();
    }
  }
  ++max_docid_;
#ifdef PERFORMANCE_TESTING
  double end = utils::getmillisecs();
  if (max_docid_ % 10000 == 0) {
    LOG(INFO) << "table cost [" << end_table - start << "]ms, vec store cost ["
              << end - end_table << "]ms";
  }
#endif
  return 0;
}

int GammaEngine::AddOrUpdateDocs(Docs &docs, BatchResult &result) {
#ifdef PERFORMANCE_TESTING
  double start = utils::getmillisecs();
#endif
  std::vector<Doc> &doc_vec = docs.GetDocs();
  int batch_size = 0, start_id = 0;

  auto batchAdd = [&](int start_id, int batch_size) {
    if (batch_size <= 0) return;

    int ret =
        table_->BatchAdd(start_id, batch_size, max_docid_, doc_vec, result);
    if (ret != 0) {
      LOG(ERROR) << "Add to table error";
      return;
    }

    for (int i = start_id; i < start_id + batch_size; ++i) {
      Doc &doc = doc_vec[i];
#ifndef BUILD_GPU
      std::vector<struct Field> &fields_table = doc.TableFields();
      for (size_t j = 0; j < fields_table.size(); ++j) {
        struct Field &field = fields_table[j];
        int idx = table_->GetAttrIdx(field.name);
        field_range_index_->Add(max_docid_ + i - start_id, idx);
      }
#endif  // BUILD_GPU
      // add vectors by VectorManager
      std::vector<struct Field> &fields_vec = doc.VectorFields();
      ret = vec_manager_->AddToStore(max_docid_ + i - start_id, fields_vec);
      if (ret != 0) {
        std::string msg = "Add to vector manager error";
        result.SetResult(i, -1, msg);
        LOG(ERROR) << msg;
        continue;
      }
    }

    max_docid_ += batch_size;
  };

  for (size_t i = 0; i < doc_vec.size(); ++i) {
    Doc &doc = doc_vec[i];
    std::string &key = doc.Key();
    // add fields into table
    int docid = -1;
    table_->GetDocIDByKey(key, docid);
    if (docid == -1) {
      ++batch_size;
      continue;
    } else {
      batchAdd(start_id, batch_size);
      batch_size = 0;
      start_id = i;
      std::vector<struct Field> &fields_table = doc.TableFields();
      std::vector<struct Field> &fields_vec = doc.VectorFields();
      if (Update(docid, fields_table, fields_vec)) {
        LOG(ERROR) << "update error, key=" << key << ", docid=" << docid;
        continue;
      }
    }
  }

  batchAdd(start_id, batch_size);
  if (not b_running_ and index_status_ == UNINDEXED) {
    if (max_docid_ >= indexing_size_) {
      LOG(INFO) << "Begin indexing.";
      this->BuildIndex();
    }
  }
#ifdef PERFORMANCE_TESTING
  double end = utils::getmillisecs();
  if (max_docid_ % 10000 == 0) {
    LOG(INFO) << "Add total cost [" << end - start << "]ms";
  }
#endif
  return 0;
}

int GammaEngine::Update(Doc *doc) { return -1; }

int GammaEngine::Update(int doc_id, std::vector<struct Field> &fields_table,
                        std::vector<struct Field> &fields_vec) {
  int ret = vec_manager_->Update(doc_id, fields_vec);
  if (ret != 0) {
    return ret;
  }

#ifndef BUILD_GPU
  for (size_t i = 0; i < fields_table.size(); ++i) {
    struct Field &field = fields_table[i];
    int idx = table_->GetAttrIdx(field.name);
    field_range_index_->Delete(doc_id, idx);
  }
#endif  // BUILD_GPU

  if (table_->Update(fields_table, doc_id) != 0) {
    LOG(ERROR) << "table update error";
    return -1;
  }

#ifndef BUILD_GPU
  for (size_t i = 0; i < fields_table.size(); ++i) {
    struct Field &field = fields_table[i];
    int idx = table_->GetAttrIdx(field.name);
    field_range_index_->Add(doc_id, idx);
  }
#endif  // BUILD_GPU

#ifdef DEBUG
  LOG(INFO) << "update success! key=" << key;
#endif
  return 0;
}

int GammaEngine::Delete(std::string &key) {
  int docid = -1, ret = 0;
  ret = table_->GetDocIDByKey(key, docid);
  if (ret != 0 || docid < 0) return -1;

  if (bitmap::test(docids_bitmap_, docid)) {
    return ret;
  }
  ++delete_num_;
  bitmap::set(docids_bitmap_, docid);
  table_->Delete(key);

  vec_manager_->Delete(docid);

  return ret;
}

int GammaEngine::DelDocByQuery(Request &request) {
#ifdef DEBUG
// LOG(INFO) << "delete by query request:" << RequestToString(request);
#endif

#ifndef BUILD_GPU
  std::vector<struct RangeFilter> &range_filters = request.RangeFilters();

  if (range_filters.size() <= 0) {
    LOG(ERROR) << "no range filter";
    return 1;
  }
  MultiRangeQueryResults range_query_result;  // Note its scope

  std::vector<FilterInfo> filters;
  filters.resize(range_filters.size());
  int idx = 0;

  for (size_t i = 0; i < range_filters.size(); ++i) {
    struct RangeFilter &range_filter = range_filters[i];

    filters[idx].field = table_->GetAttrIdx(range_filter.field);
    filters[idx].lower_value = range_filter.lower_value;
    filters[idx].upper_value = range_filter.upper_value;

    ++idx;
  }

  int retval = field_range_index_->Search(filters, &range_query_result);
  if (retval == 0) {
    LOG(ERROR) << "numeric index search error, ret=" << retval;
    return 1;
  }

  std::vector<int> doc_ids = range_query_result.ToDocs();
  for (size_t i = 0; i < doc_ids.size(); ++i) {
    int docid = doc_ids[i];
    if (bitmap::test(docids_bitmap_, docid)) {
      continue;
    }
    ++delete_num_;
    bitmap::set(docids_bitmap_, docid);
  }
#endif  // BUILD_GPU
  return 0;
}

int GammaEngine::GetDoc(std::string &key, Doc &doc) {
  int docid = -1, ret = 0;
  ret = table_->GetDocIDByKey(key, docid);
  if (ret != 0 || docid < 0) {
    LOG(INFO) << "GetDocIDbyKey [" << key << "] error!";
    return -1;
  }

  return GetDoc(docid, doc);
}

int GammaEngine::GetDoc(int docid, Doc &doc) {
  int ret = 0;
  if (bitmap::test(docids_bitmap_, docid)) {
    LOG(INFO) << "docid [" << docid << "] is deleted!";
    return -1;
  }
  std::vector<std::string> index_names;
  vec_manager_->VectorNames(index_names);

  table::DecompressStr decompress_str;
  table_->GetDocInfo(docid, doc, decompress_str);

  std::vector<std::pair<std::string, int>> vec_fields_ids;
  for (size_t i = 0; i < index_names.size(); ++i) {
    vec_fields_ids.emplace_back(std::make_pair(index_names[i], docid));
  }

  std::vector<std::string> vec;
  ret = vec_manager_->GetVector(vec_fields_ids, vec, true);
  if (ret == 0 && vec.size() == vec_fields_ids.size()) {
    for (size_t i = 0; i < index_names.size(); ++i) {
      struct Field field;
      field.name = index_names[i];
      field.datatype = DataType::VECTOR;
      field.value = vec[i];
      doc.AddField(field);
    }
  }
  return 0;
}

int GammaEngine::BuildIndex() {
  if (b_running_) {
    if (vec_manager_->Indexing() != 0) {
      LOG(ERROR) << "Create index failed!";
      return -1;
    }
    return 0;
  }
  b_running_ = true;

  auto func_indexing = std::bind(&GammaEngine::Indexing, this);
  std::thread t(func_indexing);
  t.detach();
  return 0;
}

int GammaEngine::Indexing() {
  if (vec_manager_->Indexing() != 0) {
    LOG(ERROR) << "Create index failed!";
    b_running_ = false;
    return -1;
  }

  LOG(INFO) << "vector manager indexing success!";
  int ret = 0;
  bool has_error = false;
  while (b_running_) {
    if (has_error) {
      usleep(5000 * 1000);  // sleep 5000ms
      continue;
    }
    int add_ret = vec_manager_->AddRTVecsToIndex();
    if (add_ret != 0) {
      has_error = true;
      LOG(ERROR) << "Add real time vectors to index error!";
      continue;
    }
    index_status_ = IndexStatus::INDEXED;
    usleep(1000 * 1000);  // sleep 5000ms
  }
  running_cv_.notify_one();
  LOG(INFO) << "Build index exited!";
  return ret;
}

int GammaEngine::BuildFieldIndex() {
  b_field_running_ = true;

  std::map<std::string, enum DataType> attr_type_map;
  table_->GetAttrType(attr_type_map);
  int field_num = attr_type_map.size();

  while (b_field_running_) {
    if (b_loading_) {
      usleep(5000 * 1000);  // sleep 5000ms
      continue;
    }
    int lastest_num = max_docid_;

#pragma omp parallel for
    for (int i = 0; i < field_num; ++i) {
      for (int j = indexed_field_num_; j < lastest_num; ++j) {
        // field_range_index_->Add(j, i);
        ;
      }
    }

    indexed_field_num_ = lastest_num;
    usleep(5000 * 1000);  // sleep 5000ms
  }
  running_field_cv_.notify_one();
  LOG(INFO) << "Build field index exited!";
  return 0;
}

int GammaEngine::GetDocsNum() { return max_docid_ - delete_num_; }

void GammaEngine::GetIndexStatus(EngineStatus &engine_status) {
  engine_status.SetIndexStatus(index_status_);

  long table_mem_bytes = table_->GetMemoryBytes();
  long vec_mem_bytes, index_mem_bytes;
  vec_manager_->GetTotalMemBytes(index_mem_bytes, vec_mem_bytes);

  long dense_b = 0, sparse_b = 0, total_mem_b = 0;
#ifndef BUILD_GPU

  if (field_range_index_) {
    total_mem_b += field_range_index_->MemorySize(dense_b, sparse_b);
  }

  // long total_mem_kb = total_mem_b / 1024;
  // long total_mem_mb = total_mem_kb / 1024;
  // LOG(INFO) << "Field range memory [" << total_mem_kb << "]kb, ["
  //           << total_mem_mb << "]MB, dense [" << dense_b / 1024 / 1024
  //           << "]MB sparse [" << sparse_b / 1024 / 1024
  //           << "]MB, indexed_field_num_ [" << indexed_field_num_ << "]";
#endif  // BUILD_GPU

  engine_status.SetTableMem(table_mem_bytes);
  engine_status.SetIndexMem(index_mem_bytes);
  engine_status.SetVectorMem(vec_mem_bytes);
  engine_status.SetFieldRangeMem(total_mem_b);
  engine_status.SetBitmapMem(bitmap_bytes_size_);
  engine_status.SetDocNum(GetDocsNum());
  engine_status.SetMaxDocID(max_docid_ - 1);
}

int GammaEngine::Dump() {
  int max_docid = max_docid_ - 1;
  if (max_docid <= dump_docid_) {
    LOG(INFO) << "No fresh doc, cannot dump.";
    return 0;
  }

  std::time_t t = std::time(nullptr);
  char tm_str[100];
  std::strftime(tm_str, sizeof(tm_str), date_time_format_.c_str(),
                std::localtime(&t));

  string path = dump_path_ + "/" + tm_str;
  if (!utils::isFolderExist(path.c_str())) {
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  int ret = table_io_->Dump(0, max_docid + 1);
  if (ret != 0) {
    LOG(ERROR) << "dump table error, ret=" << ret;
    return -1;
  }
  ret = vec_manager_->Dump(path, 0, max_docid);
  if (ret != 0) {
    LOG(ERROR) << "dump vector error, ret=" << ret;
    return -1;
  }

  const string bp_name = path + "/" + "bitmap";
  FILE *fp_output = fopen(bp_name.c_str(), "wb");
  if (fp_output == nullptr) {
    LOG(ERROR) << "Cannot write file " << bp_name;
    return -1;
  }

  if ((size_t)bitmap_bytes_size_ != fwrite((void *)(docids_bitmap_),
                                           sizeof(char), bitmap_bytes_size_,
                                           fp_output)) {
    LOG(ERROR) << "write bitmap error";
    return -2;
  }
  fclose(fp_output);

  const string dump_done_file = path + "/dump.done";
  std::ofstream f_done;
  f_done.open(dump_done_file);
  if (!f_done.is_open()) {
    LOG(ERROR) << "Cannot create file " << dump_done_file;
    return -1;
  }
  f_done << "start_docid " << 0 << std::endl;
  f_done << "end_docid " << max_docid << std::endl;
  f_done.close();

  if (last_dump_dir_ != "" && utils::remove_dir(last_dump_dir_.c_str())) {
    LOG(ERROR) << "remove last dump directory error, path=" << last_dump_dir_;
  }
  dump_docid_ = max_docid + 1;
  LOG(INFO) << "Dumped to [" << path << "], next dump docid [" << dump_docid_
            << "], last dump directory(removed)=" << last_dump_dir_;
  last_dump_dir_ = path;
  return 0;
}

int GammaEngine::CreateTableFromLocal(std::string &table_name) {
  std::vector<string> file_paths = utils::ls(index_root_path_);
  for (string &file_path : file_paths) {
    std::string::size_type pos = file_path.rfind(".schema");
    if (pos == file_path.size() - 7) {
      std::string::size_type begin = file_path.rfind('/');
      assert(begin != std::string::npos);
      begin += 1;
      table_name = file_path.substr(begin, pos - begin);
      LOG(INFO) << "local table name=" << table_name;
      TableSchemaIO tio(file_path);
      TableInfo table;
      if (tio.Read(table_name, table)) {
        LOG(ERROR) << "read table schema error, path=" << file_path;
        return -1;
      }

      if (CreateTable(table)) {
        LOG(ERROR) << "create table error when loading";
        return -1;
      }
      return 0;
    }
  }
  return -1;
}

int GammaEngine::Load() {
  b_loading_ = true;
  if (!created_table_) {
    string table_name;
    if (CreateTableFromLocal(table_name)) {
      LOG(ERROR) << "create table from local error";
      return -1;
    }
    LOG(INFO) << "create table from local success, table name=" << table_name;
  }
  af_exector_->Stop();

  std::vector<std::pair<std::time_t, string>> folders_tm;
  std::vector<string> folders = utils::ls_folder(dump_path_);
  std::vector<string> folders_not_done;
  for (const string &folder_name : folders) {
    if (folder_name == "") continue;
    string folder_path = dump_path_ + "/" + folder_name;
    string done_file = folder_path + "/dump.done";
    if (!utils::file_exist(done_file)) {
      LOG(INFO) << "done file is not existed, skip it! path=" << done_file;
      folders_not_done.push_back(folder_path);
      continue;
    }
    struct tm result;
    strptime(folder_name.c_str(), date_time_format_.c_str(), &result);
    std::time_t t = std::mktime(&result);
    folders_tm.push_back(std::make_pair(t, folder_path));
  }
  std::sort(folders_tm.begin(), folders_tm.end(),
            [](const std::pair<std::time_t, string> &a,
               const std::pair<std::time_t, string> &b) {
              return a.first < b.first;
            });

  int ret = 0;
  std::vector<string> dirs;
  ret = table_io_->Load(max_docid_);
  if (ret != 0) {
    LOG(ERROR) << "load profile error, ret=" << ret;
    return ret;
  }

#ifndef BUILD_GPU
  int field_num = table_->FieldsNum();
  for (int i = 0; i < max_docid_; ++i) {
    for (int j = 0; j < field_num; ++j) {
      field_range_index_->Add(i, j);
    }
  }
#endif

  string last_dir = "";
  if (folders_tm.size() > 0) {
    last_dir = folders_tm[folders_tm.size() - 1].second;
    LOG(INFO) << "Loading from " << last_dir;
    dirs.push_back(last_dir);
    // load bitmap
    string bitmap_file_name = last_dir + "/bitmap";
    FILE *fp_bm = fopen(bitmap_file_name.c_str(), "rb");
    if (fp_bm == nullptr) {
      LOG(ERROR) << "Cannot open file " << bitmap_file_name;
      return IO_ERR;
    }
    long bm_file_size = utils::get_file_size(bitmap_file_name.c_str());
    fread((void *)(docids_bitmap_), sizeof(char), bm_file_size, fp_bm);
    fclose(fp_bm);
    // compatiable with v3.1.0
    assert(bm_file_size <= bitmap_bytes_size_);

    delete_num_ = 0;
    for (int i = 0; i < max_docid_; ++i) {
      if (bitmap::test(docids_bitmap_, i)) {
        ++delete_num_;
      }
    }
  }
  ret = vec_manager_->Load(dirs, max_docid_);
  if (ret != 0) {
    LOG(ERROR) << "load vector error, ret=" << ret << ", path=" << last_dir;
    return ret;
  }
  if (not b_running_ and index_status_ == UNINDEXED) {
    if (max_docid_ >= indexing_size_) {
      LOG(INFO) << "Begin indexing.";
      this->BuildIndex();
    }
  }
  // remove directorys which are not done
  for (const string &folder : folders_not_done) {
    if (utils::remove_dir(folder.c_str())) {
      LOG(ERROR) << "clean error, not done directory=" << folder;
    }
  }

  dump_docid_ = max_docid_;
  last_dump_dir_ = last_dir;
  af_exector_->Start();
  LOG(INFO) << "load engine success! max docid=" << max_docid_
            << ", load directory=" << last_dir
            << ", clean directorys(not done)="
            << utils::join(folders_not_done, ',');
  b_loading_ = false;
  return 0;
}

int GammaEngine::AddNumIndexFields() {
  int retvals = 0;
  std::map<std::string, enum DataType> attr_type;
  retvals = table_->GetAttrType(attr_type);

  std::map<std::string, bool> attr_index;
  retvals = table_->GetAttrIsIndex(attr_index);
  for (const auto &it : attr_type) {
    string field_name = it.first;
    const auto &attr_index_it = attr_index.find(field_name);
    if (attr_index_it == attr_index.end()) {
      LOG(ERROR) << "Cannot find field [" << field_name << "]";
      continue;
    }
    bool is_index = attr_index_it->second;
    if (not is_index) {
      continue;
    }
    int field_idx = table_->GetAttrIdx(field_name);
    LOG(INFO) << "Add range field [" << field_name << "]";
    field_range_index_->AddField(field_idx, it.second);
  }
  return retvals;
}

int GammaEngine::PackResults(const GammaResult *gamma_results,
                             Response &response_results, Request &request) {
  for (int i = 0; i < request.ReqNum(); ++i) {
    struct SearchResult result;
    result.total = gamma_results[i].total;

    auto string_field_num = table_->StringFieldNum();
    if (table_->IsCompress() && string_field_num > 0) {
      std::vector<std::vector<int> *> doc_bucket(MAX_SEGMENT_NUM);
      std::fill(doc_bucket.begin(), doc_bucket.end(), nullptr);

      for (int j = 0; j < gamma_results[i].results_count; ++j) {
        int docid = gamma_results[i].docs[j]->docid;
        int bucket_id = docid / DOCNUM_PER_SEGMENT;
        if (doc_bucket[bucket_id] == nullptr) {
          doc_bucket[bucket_id] = new std::vector<int>;
        }
        doc_bucket[bucket_id]->push_back(j);
      }

      result.result_items.resize(gamma_results[i].results_count);

      for (int j = 0; j < MAX_SEGMENT_NUM; ++j) {
        if (doc_bucket[j] == nullptr) {
          continue;
        }

        std::vector<int> *bucket = doc_bucket[j];
        table::DecompressStr decompress_str;
        for (int k = 0; k < bucket->size(); ++k) {
          int idx = (*bucket)[k];
          VectorDoc *vec_doc = gamma_results[i].docs[idx];
          struct ResultItem result_item;
          PackResultItem(vec_doc, request, result_item, decompress_str);
          result.result_items[idx] = std::move(result_item);
        }
        delete bucket;
      }
    } else {
      result.result_items.resize(gamma_results[i].results_count);
      table::DecompressStr decompress_str;
      for (int j = 0; j < gamma_results[i].results_count; ++j) {
        int docid = gamma_results[i].docs[j]->docid;
        VectorDoc *vec_doc = gamma_results[i].docs[j];
        struct ResultItem result_item;
        PackResultItem(vec_doc, request, result_item, decompress_str);
        result.result_items[j] = std::move(result_item);
      }
    }
    result.msg = "Success";
    result.result_code = SearchResultCode::SUCCESS;
    response_results.AddResults(std::move(result));
  }
  return 0;
}

int GammaEngine::PackResultItem(const VectorDoc *vec_doc, Request &request,
                                struct ResultItem &result_item,
                                table::DecompressStr &decompress_str) {
  result_item.score = vec_doc->score;

  Doc doc;
  int docid = vec_doc->docid;

  std::vector<std::string> &vec_fields = request.Fields();

  // add vector into result
  size_t fields_size = vec_fields.size();
  if (fields_size != 0) {
    std::vector<std::pair<string, int>> vec_fields_ids;
    std::vector<string> table_fields;

    for (size_t i = 0; i < fields_size; ++i) {
      std::string &name = vec_fields[i];
      const auto index = vec_manager_->GetVectorIndex(name);
      if (index == nullptr) {
        table_fields.push_back(name);
      } else {
        vec_fields_ids.emplace_back(std::make_pair(name, docid));
      }
    }

    std::vector<string> vec;
    int ret = vec_manager_->GetVector(vec_fields_ids, vec, true);

    int table_fields_num = 0;

    if (table_fields.size() == 0) {
      table_fields_num = table_->FieldsNum();

      table_->GetDocInfo(docid, doc, decompress_str);
    } else {
      table_fields_num = table_fields.size();

      for (int i = 0; i < table_fields_num; ++i) {
        struct tig_gamma::Field field;
        table_->GetFieldInfo(docid, table_fields[i], field, decompress_str);
        doc.AddField(std::move(field));
      }
    }

    if (ret == 0 && vec.size() == vec_fields_ids.size()) {
      for (size_t i = 0; i < vec_fields_ids.size(); ++i) {
        const string &field_name = vec_fields_ids[i].first;
        result_item.names.emplace_back(std::move(field_name));
        result_item.values.emplace_back(vec[i]);
      }
    } else {
      // get vector error
      // TODO : release extra field
      ;
    }
  } else {
    table_->GetDocInfo(docid, doc, decompress_str);
  }

  std::vector<struct Field> &fields = doc.TableFields();
  result_item.names.resize(fields.size());
  result_item.values.resize(fields.size());

  int i = 0;
  for (struct Field &field : fields) {
    result_item.names[i] = std::move(field.name);
    result_item.values[i] = std::move(field.value);
    ++i;
  }

  cJSON *extra_json = cJSON_CreateObject();
  cJSON *vec_result_json = cJSON_CreateArray();
  cJSON_AddItemToObject(extra_json, EXTRA_VECTOR_RESULT.c_str(),
                        vec_result_json);
  for (int i = 0; i < vec_doc->fields_len; ++i) {
    VectorDocField *vec_field = vec_doc->fields + i;
    cJSON *vec_field_json = cJSON_CreateObject();

    cJSON_AddStringToObject(vec_field_json, EXTRA_VECTOR_FIELD_NAME.c_str(),
                            vec_field->name.c_str());
    string source = string(vec_field->source, vec_field->source_len);
    cJSON_AddStringToObject(vec_field_json, EXTRA_VECTOR_FIELD_SOURCE.c_str(),
                            source.c_str());
    cJSON_AddNumberToObject(vec_field_json, EXTRA_VECTOR_FIELD_SCORE.c_str(),
                            vec_field->score);
    cJSON_AddItemToArray(vec_result_json, vec_field_json);
  }

  char *extra_data = cJSON_PrintUnformatted(extra_json);
  result_item.extra = std::string(extra_data, std::strlen(extra_data));
  free(extra_data);
  cJSON_Delete(extra_json);

  return 0;
}

}  // namespace tig_gamma
