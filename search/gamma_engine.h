/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <string>

#include "api_data/gamma_batch_result.h"
#include "api_data/gamma_doc.h"
#include "api_data/gamma_docs.h"
#include "api_data/gamma_engine_status.h"
#include "api_data/gamma_request.h"
#include "api_data/gamma_response.h"
#include "api_data/gamma_table.h"
#include "async_flush.h"
#include "field_range_index.h"
#include "table.h"
#include "table_io.h"
#include "vector_manager.h"

namespace tig_gamma {

enum IndexStatus { UNINDEXED = 0, INDEXING, INDEXED };

class GammaEngine {
 public:
  static GammaEngine *GetInstance(const std::string &index_root_path);

  ~GammaEngine();

  int Setup();

  int Search(Request &request, Response &response_results);

  int CreateTable(TableInfo &table);

  int AddOrUpdate(Doc &doc);

  int AddOrUpdateDocs(Docs &docs, BatchResult &result);

  int Update(Doc *doc);
  int Update(int doc_id, std::vector<struct Field> &fields_table,
             std::vector<struct Field> &fields_vec);

  /**
   * Delete doc
   * @param key
   * @return 0 if successed
   */
  int Delete(std::string &key);

  /**
   * Delete doc by query
   * @param request delete request
   * @return 0 if successed
   */
  int DelDocByQuery(Request &request);

  int GetDoc(std::string &key, Doc &doc);

  int GetDoc(int docid, Doc &doc);

  /**
   * blocking to build index
   * @return 0 if exited
   */
  int BuildIndex();
  int BuildFieldIndex();

  void GetIndexStatus(EngineStatus &engine_status);

  int Dump();

  int Load();

  int GetDocsNum();

  table::Table *GetTable() { return table_; }

  VectorManager *GetVectorManager() { return vec_manager_; }

  int SetBatchDocsNum(int i) {
    batch_docs_.resize(i);
    return 0;
  }

  int BatchDocsPrepare(char *doc_str, int idx) {
    if (idx >= batch_docs_.size()) {
      LOG(ERROR) << "idx [" << idx << "] > batch_docs size ["
                 << batch_docs_.size() << "]";
      return -1;
    }
    batch_docs_[idx] = doc_str;
    return 0;
  }

  char **BatchDocsStr() { return batch_docs_.data(); }

  int GetConfig(Config &config);

  int SetConfig(Config &config);

 private:
  GammaEngine(const std::string &index_root_path);

  int CreateTableFromLocal(std::string &table_name);

  int Indexing();

 private:
  std::string index_root_path_;
  std::string dump_path_;

  MultiFieldsRangeIndex *field_range_index_;

  char *docids_bitmap_;
  table::Table *table_;
  VectorManager *vec_manager_;

  int AddNumIndexFields();

  int max_docid_;
  int indexing_size_;

  std::atomic<int> delete_num_;

  int b_running_; // 0 not run, not 0 running
  bool b_field_running_;

  std::condition_variable running_cv_;
  std::condition_variable running_field_cv_;

  int PackResults(const GammaResult *gamma_results, Response &response_results,
                  Request &request);

  int PackResultItem(const VectorDoc *vec_doc, Request &request,
                     struct ResultItem &result_item,
                     table::DecompressStr &decompress_str);

  int MultiRangeQuery(Request &request, GammaSearchCondition *condition,
                      Response &response_results,
                      MultiRangeQueryResults *range_query_result);

  enum IndexStatus index_status_;

  int dump_docid_;  // next dump docid
  int bitmap_bytes_size_;
  const std::string date_time_format_;
  std::string last_dump_dir_;  // it should be delete after next dump

  bool created_table_;

  bool b_loading_;

  std::vector<char *> batch_docs_;

#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_num_;
#endif

  TableIO *table_io_;
  AsyncFlushExecutor *af_exector_;
};


class RequestConcurrentController {
 public:
  static RequestConcurrentController &GetInstance() {
    static RequestConcurrentController intance;
    return intance;
  }

  ~RequestConcurrentController() = default;

  bool Acquire(int req_num);

  void Release(int req_num);

 private:
  RequestConcurrentController();

  RequestConcurrentController(const RequestConcurrentController &) = delete;

  RequestConcurrentController &operator=(const RequestConcurrentController &) =
      delete;

  int GetMaxThread();

  int GetSystemInfo(const char *cmd);

 private:
  int cur_concurrent_num_;
  int concurrent_threshold_;
  int max_threads_;
};

}  // namespace tig_gamma
