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

#include "api_data/gamma_doc.h"
#include "api_data/gamma_table.h"
#include "log.h"
#include "table_data.h"
#include "table_define.h"

#ifdef USE_BTREE
#include "threadskv10h.h"
#endif

#ifdef WITH_ROCKSDB
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#endif

using namespace table;

namespace table {

/** table, support add, update, delete, dump and load.
 */
class Table {
 public:
  explicit Table(const std::string &root_path);

  ~Table();

  /** create table
   *
   * @param table  table definition
   * @return 0 if successed
   */
  int CreateTable(tig_gamma::TableInfo &table);

  /** add a doc to table
   *
   * @param doc     doc to add
   * @param docid   doc index number
   * @return 0 if successed
   */
  int Add(const std::vector<struct tig_gamma::Field> &fields, int docid);

  /** update a doc
   *
   * @param doc     doc to update
   * @param docid   doc index number
   * @return 0 if successed
   */
  int Update(const std::vector<struct tig_gamma::Field> &fields, int docid);

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
  int Dump(const std::string &path, int start_docid, int end_docid);

  long GetMemoryBytes();

  int GetDocInfo(std::string &id, tig_gamma::Doc &doc, DecompressStr &decompress_str);
  int GetDocInfo(const int docid, tig_gamma::Doc &doc, DecompressStr &decompress_str);

  void GetFieldInfo(const int docid, const std::string &field_name,
                    struct tig_gamma::Field &field, DecompressStr &decompress_str);

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

  int GetFieldType(const std::string &field, tig_gamma::DataType &type);

  int GetAttrType(std::map<std::string, tig_gamma::DataType> &attr_type_map);

  int GetAttrIsIndex(std::map<std::string, bool> &attr_is_index_map);

  int GetAttrIdx(const std::string &field) const;

  int Load(const std::vector<std::string> &folders, int &doc_num);

  int FieldsNum() { return attrs_.size(); };

 private:
  int FTypeSize(tig_gamma::DataType fType);

  void SetFieldValue(int docid, const std::string &field, int field_id,
                     const char *value, str_len_t len);

  int AddField(const std::string &name, tig_gamma::DataType ftype,
               bool is_index);

  void ToRowKey(int id, std::string &key) const;

  int GetRawDoc(int docid, std::vector<char> &raw_doc);

  int GetSegPos(IN int32_t docid, IN int32_t field_id, OUT int &seg_pos,
                OUT size_t &in_seg_pos);

  int PutToDB(int docid);

  int Extend();

  std::string name_;   // table name
  int item_length_;    // every doc item length
  uint8_t field_num_;  // field number
  int key_idx_;        // key postion

  std::map<int, std::string> idx_attr_map_;
  std::map<std::string, int> attr_idx_map_;
  std::map<std::string, tig_gamma::DataType> attr_type_map_;
  std::map<std::string, bool> attr_is_index_map_;
  std::vector<int> idx_attr_offset_;
  std::vector<tig_gamma::DataType> attrs_;

  uint8_t id_type_;  // 0 string, 1 long, default 1
  cuckoohash_map<long, int> item_to_docid_;
  cuckoohash_map<std::string, int> item_to_docid_str_;

  TableData *main_file_[MAX_SEGMENT_NUM];
  int seg_num_;  // cur segment num

  bool table_created_;
#ifdef WITH_ROCKSDB
  rocksdb::DB *db_;
#endif

#ifdef USE_BTREE
  BtMgr *main_mgr_;
  BtMgr *cache_mgr_;
#endif
  std::string db_path_;
};

}  // namespace table