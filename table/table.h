/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuckoohash_map.hh>
#include <map>
#include <string>
#include <vector>

#include "api_data/gamma_batch_result.h"
#include "api_data/gamma_doc.h"
#include "api_data/gamma_table.h"
#include "io_common.h"
#include "log.h"
#include "table_data.h"
#include "table_define.h"

#ifdef USE_BTREE
#include "threadskv10h.h"
#endif

using namespace tig_gamma::table;

namespace tig_gamma {

namespace table {

class TableIO;

struct TableParams : DumpConfig {
  // currently no configure need to dump
  TableParams(const std::string &name_ = "") : DumpConfig(name_) {}
  int Parse(utils::JsonParser &jp) { return 0; }
};

/** table, support add, update, delete, dump and load.
 */
class Table {
 public:
  explicit Table(const std::string &root_path, bool b_compress = false);

  ~Table();

  /** create table
   *
   * @param table  table definition
   * @param table_params unused
   * @return 0 if successed
   */
  int CreateTable(TableInfo &table, TableParams &table_params);

  /** add a doc to table
   *
   * @param key     doc's key
   * @param doc     doc to add
   * @param docid   doc index number
   * @return 0 if successed
   */
  int Add(const std::string &key, const std::vector<struct Field> &fields,
          int docid);

  int BatchAdd(int start_id, int batch_size, int docid,
               std::vector<Doc> &doc_vec, BatchResult &result);

  /** update a doc
   *
   * @param doc     doc to update
   * @param docid   doc index number
   * @return 0 if successed
   */
  int Update(const std::vector<struct Field> &fields, int docid);

  int Delete(std::string &key);

  /** get docid by key
   *
   * @param key key to get
   * @param docid output, the docid to key
   * @return 0 if successed, -1 key not found
   */
  int GetDocIDByKey(std::string &key, int &docid);

  /** dump datas to disk
   *
   * @return ResultCode
   */
  // int Dump(const std::string &path, int start_docid, int end_docid);

  long GetMemoryBytes();

  int GetDocInfo(std::string &id, Doc &doc, DecompressStr &decompress_str);
  int GetDocInfo(const int docid, Doc &doc, DecompressStr &decompress_str);

  void GetFieldInfo(const int docid, const std::string &field_name,
                    struct Field &field, DecompressStr &decompress_str);

  template <typename T>
  bool GetField(const int docid, const int field_id, T &value) {
    if ((docid < 0) or (field_id < 0 || field_id >= field_num_)) return false;

    size_t offset = idx_attr_offset_[field_id];

    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, field_id, seg_pos, in_seg_pos);
    if (ret != 0) {
      return false;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    char *base = seg_file->Base();
    memcpy(&value, base + offset, sizeof(T));
    return true;
  }

  template <typename T>
  void GetField(int docid, const std::string &field, T &value) {
    const auto &iter = attr_idx_map_.find(field);
    if (iter == attr_idx_map_.end()) {
      return;
    }
    GetField<T>(docid, iter->second, value);
  }

  int GetFieldString(int docid, const std::string &field, std::string &value,
                     DecompressStr &decompress_str);

  int GetFieldString(int docid, int field_id, std::string &value,
                     DecompressStr &decompress_str);

  int GetFieldRawValue(int docid, int field_id, std::string &value);

  int GetFieldType(const std::string &field, DataType &type);

  int GetAttrType(std::map<std::string, DataType> &attr_type_map);

  int GetAttrIsIndex(std::map<std::string, bool> &attr_is_index_map);

  int GetAttrIdx(const std::string &field) const;

  uint8_t StringFieldNum() const { return string_field_num_; }

  int Load(int &doc_num);

  int Sync();

  int FieldsNum() { return attrs_.size(); }

  std::map<std::string, int> &FieldMap() { return attr_idx_map_; }

  DumpConfig *GetDumpConfig() { return table_params_; }

  int GetRawDoc(int docid, std::vector<char> &raw_doc);

  bool IsCompress() { return b_compress_; }

  std::string root_path_;
  int last_docid_;

 private:
  int FTypeSize(DataType fType);

  void SetFieldValue(int docid, const std::string &field, int field_id,
                     const char *value, str_len_t len);

  int AddField(const std::string &name, DataType ftype, bool is_index);

  // void ToRowKey(int id, std::string &key) const;

  int AddRawDoc(int docid, const char *raw_doc, int doc_size);

  int GetSegPos(IN int32_t docid, IN int32_t field_id, OUT int &seg_pos,
                OUT size_t &in_seg_pos, bool bRead = true);

  // int PutToDB(int docid);

  int BatchPutToDB(int docid, int batch_size);

  int Extend();

  void Compress();

  void BufferQueueWorker();

  std::string name_;   // table name
  int item_length_;    // every doc item length
  uint8_t field_num_;  // field number
  uint8_t string_field_num_;
  int key_idx_;  // key postion

  std::map<int, std::string> idx_attr_map_;
  std::map<std::string, int> attr_idx_map_;
  std::map<std::string, DataType> attr_type_map_;
  std::map<std::string, bool> attr_is_index_map_;
  std::vector<int> idx_attr_offset_;
  std::vector<DataType> attrs_;

  uint8_t id_type_;  // 0 string, 1 long, default 1
  bool b_compress_;
  cuckoohash_map<long, int> item_to_docid_;

  TableData *main_file_[MAX_SEGMENT_NUM];
  int seg_num_;  // cur segment num
  int compressed_num_;

  bool table_created_;

#ifdef USE_BTREE
  BtMgr *main_mgr_;
  BtMgr *cache_mgr_;
#endif
  TableParams *table_params_;
};

}  // namespace table
}  // namespace tig_gamma
