/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef GAMMA_ENGINE_H_
#define GAMMA_ENGINE_H_

#include "field_range_index.h"
#include "gamma_api.h"
#include "profile.h"
#include "vector_manager.h"

#include <condition_variable>
#include <string>

namespace tig_gamma {

class GammaEngine {
 public:
  static GammaEngine *GetInstance(const std::string &index_root_path,
                                  int max_doc_size);

  ~GammaEngine();

  int Setup(int max_doc_size);

  Response *Search(const Request *request);

  int CreateTable(const Table *table);

  int Add(const Doc *doc);
  int AddOrUpdate(const Doc *doc);

  int Update(const Doc *doc);
  int Update(int doc_id, std::vector<Field *> &fields_profile,
             std::vector<Field *> &fields_vec);

  /**
   * Delete doc
   * @param key
   * @return 0 if successed
   */
  int Del(ByteArray *key);

  /**
   * Delete doc by query
   * @param request delete request
   * @return 0 if successed
   */
  int DelDocByQuery(Request *request);

  Doc *GetDoc(ByteArray *id);

  /**
   * blocking to build index
   * @return 0 if exited
   */
  int BuildIndex();
  int BuildFieldIndex();

  int GetIndexStatus();

  int Dump();

  int Load();

  int GetDocsNum();

  long GetMemoryBytes();

 private:
  GammaEngine(const std::string &index_root_path);
  int CreateTableFromLocal(std::string &table_name);

  int Indexing();

 private:
  std::string index_root_path_;
  std::string dump_path_;

  MultiFieldsRangeIndex *field_range_index_;

  char *docids_bitmap_;
  Profile *profile_;
  VectorManager *vec_manager_;

  int AddNumIndexFields();
  template <typename T>
  int AddNumIndexField(const std::string &field);

  int max_docid_;
  int max_doc_size_;

  std::atomic<int> delete_num_;

  bool b_running_;
  bool b_field_running_;

  std::condition_variable running_cv_;
  std::condition_variable running_field_cv_;

  int PackResults(const GammaResult *gamma_results, Response *response_results,
                  const Request *request);

  ResultItem *PackResultItem(const VectorDoc *vec_doc, const Request *request);

  int MultiRangeQuery(const Request *request, GammaSearchCondition &condition,
                      Response *response_results,
                      MultiRangeQueryResults *range_query_result,
                      utils::OnlineLogger &logger);

  enum IndexStatus index_status_;

  int dump_docid_;  // next dump docid
  int bitmap_bytes_size_;
  const std::string date_time_format_;
  std::string last_bitmap_filename_; // it should be delete after next dump

  bool created_table_;
  string dump_backup_path_;

  int indexed_field_num_;

  bool b_loading_;

#ifdef PERFORMANCE_TESTING
  std::atomic<uint64_t> search_num_;
#endif

  GammaCounters *counters_;
};

// specialization for string
template <>
int GammaEngine::AddNumIndexField<std::string>(const std::string &field);

}  // namespace tig_gamma
#endif
