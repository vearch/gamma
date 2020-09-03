/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef GAMMA_ENGINE_H_
#define GAMMA_ENGINE_H_

#include <condition_variable>
#include <string>

#include "api_data/gamma_doc.h"
#include "api_data/gamma_engine_status.h"
#include "api_data/gamma_request.h"
#include "api_data/gamma_response.h"
#include "api_data/gamma_table.h"
#include "field_range_index.h"
#include "table.h"
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

  int Add(Doc *doc);
  int AddOrUpdate(Doc &doc);

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

 private:
  GammaEngine(const std::string &index_root_path);
  int CreateTableFromLocal(std::string &table_name);
  int Indexing();

 private:
  std::string index_root_path_;
  std::string dump_path_;

  MultiFieldsRangeIndex *field_range_index_;

  char *docids_bitmap_;
  Table *table_;
  VectorManager *vec_manager_;

  int AddNumIndexFields();
  template <typename T>
  int AddNumIndexField(const std::string &field);

  int max_docid_;
  int indexing_size_;

  std::atomic<int> delete_num_;

  bool b_running_;
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
                      MultiRangeQueryResults *range_query_result,
                      utils::OnlineLogger &logger);

  enum IndexStatus index_status_;

  int dump_docid_;  // next dump docid
  int bitmap_bytes_size_;
  const std::string date_time_format_;
  std::string last_dump_dir_;  // it should be delete after next dump

  bool created_table_;

  int indexed_field_num_;

  bool b_loading_;

#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_num_;
#endif
};

// specialization for string
template <>
int GammaEngine::AddNumIndexField<std::string>(const std::string &field);

}  // namespace tig_gamma
#endif
