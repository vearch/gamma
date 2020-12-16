/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "table.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <string>

#include "utils.h"

using std::move;
using std::string;
using std::vector;

namespace tig_gamma {
namespace table {

const static string kTableDumpedNum = "profile_dumped_num";

Table::Table(const string &root_path, bool b_compress) {
  item_length_ = 0;
  field_num_ = 0;
  string_field_num_ = 0;
  key_idx_ = -1;
  root_path_ = root_path + "/table";
  seg_num_ = 0;
  b_compress_ = b_compress;
  compressed_num_ = 0;

  // TODO : there is a failure.
  // if (!item_to_docid_.reserve(max_doc_size)) {
  //   LOG(ERROR) << "item_to_docid reserve failed, max_doc_size [" <<
  //   max_doc_size
  //              << "]";
  // }

  table_created_ = false;
  last_docid_ = 0;
  table_params_ = nullptr;
  LOG(INFO) << "Table created success!";
}

Table::~Table() {
#ifdef USE_BTREE
  if (cache_mgr_) {
    bt_mgrclose(cache_mgr_);
    cache_mgr_ = nullptr;
  }
  if (main_mgr_) {
    bt_mgrclose(main_mgr_);
    main_mgr_ = nullptr;
  }
#endif

  for (int i = 0; i < seg_num_; ++i) {
    delete main_file_[i];
  }
  CHECK_DELETE(table_params_);
  LOG(INFO) << "Table deleted.";
}

int Table::Load(int &num) {
  std::string file_name =
      root_path_ + "/" + std::to_string(seg_num_) + ".profile";
  int doc_num = 0;
  while (utils::file_exist(file_name)) {
    main_file_[seg_num_] = new TableData(item_length_);
    main_file_[seg_num_]->Load(seg_num_, root_path_);
    doc_num += main_file_[seg_num_]->Size();
    ++seg_num_;
    if (doc_num >= num) {
      doc_num = num;
      break;
    }
    file_name = root_path_ + "/" + std::to_string(seg_num_) + ".profile";
  }

  const string str_id = "_id";
  const auto &iter = attr_idx_map_.find(str_id);
  if (iter == attr_idx_map_.end()) {
    LOG(ERROR) << "cannot find field [" << str_id << "]";
    return -1;
  }

  int idx = iter->second;
#pragma omp parallel for
  for (int i = 0; i < doc_num; ++i) {
    if (id_type_ == 0) {
      std::string key;
      DecompressStr decompress_str;
      GetFieldString(i, idx, key, decompress_str);
      int64_t k = utils::StringToInt64(key);
      item_to_docid_.insert(k, i);
    } else {
      long key = -1;
      GetField<long>(i, idx, key);
      item_to_docid_.insert(key, i);
    }
  }

  LOG(INFO) << "Table load successed! doc num=" << doc_num;
  last_docid_ = doc_num;
  return 0;
}

int Table::Sync() { return 0; }

int Table::CreateTable(TableInfo &table, TableParams &table_params) {
  if (table_created_) {
    return -10;
  }
  name_ = table.Name();
  std::vector<struct FieldInfo> &fields = table.Fields();

  b_compress_ = table.IsCompress();
  LOG(INFO) << "Table compress [" << b_compress_ << "]";

  size_t fields_num = fields.size();
  for (size_t i = 0; i < fields_num; ++i) {
    const string name = fields[i].name;
    DataType ftype = fields[i].data_type;
    bool is_index = fields[i].is_index;
    LOG(INFO) << "Add field name [" << name << "], type [" << (int)ftype
              << "], index [" << is_index << "]";
    int ret = AddField(name, ftype, is_index);
    if (ret != 0) {
      return ret;
    }
  }

  if (key_idx_ == -1) {
    LOG(ERROR) << "No field _id! ";
    return -1;
  }

  if (!utils::isFolderExist(root_path_.c_str())) {
    mkdir(root_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

#ifdef USE_BTREE
  uint mainleafxtra = 0;
  uint maxleaves = 1000000;
  uint poolsize = 500;
  uint leafxtra = 0;
  uint mainpool = 500;
  uint mainbits = 16;
  uint bits = 16;

  string cache_file = root_path_ + string("/cache_") + ".dis";
  string main_file = root_path_ + string("/main_") + ".dis";

  remove(cache_file.c_str());
  remove(main_file.c_str());

  cache_mgr_ =
      bt_mgr(const_cast<char *>(cache_file.c_str()), bits, leafxtra, poolsize);
  cache_mgr_->maxleaves = maxleaves;
  main_mgr_ = bt_mgr(const_cast<char *>(main_file.c_str()), mainbits,
                     mainleafxtra, mainpool);
  main_mgr_->maxleaves = maxleaves;
#endif

  table_params_ = new TableParams("table");
  table_created_ = true;
  LOG(INFO) << "Create table " << name_
            << " success! item length=" << item_length_
            << ", field num=" << (int)field_num_;
  return 0;
}

int Table::FTypeSize(DataType fType) {
  int length = 0;
  if (fType == DataType::INT) {
    length = sizeof(int32_t);
  } else if (fType == DataType::LONG) {
    length = sizeof(int64_t);
  } else if (fType == DataType::FLOAT) {
    length = sizeof(float);
  } else if (fType == DataType::DOUBLE) {
    length = sizeof(double);
  } else if (fType == DataType::STRING) {
    length = sizeof(str_offset_t) + sizeof(str_len_t);
  }
  return length;
}

void Table::SetFieldValue(int docid, const std::string &field, int field_id,
                          const char *value, str_len_t len) {
  size_t offset = idx_attr_offset_[field_id];
  DataType attr = attrs_[field_id];

  if (attr != DataType::STRING) {
    int type_size = FTypeSize(attr);

    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos, false);
    if (ret != 0) {
      return;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    seg_file->Write(value, offset, type_size);
  } else {
    size_t ofst = sizeof(str_offset_t);
    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos, false);
    if (ret != 0) {
      return;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    str_offset_t str_offset = seg_file->StrOffset();
    seg_file->Write((char *)&str_offset, offset, sizeof(str_offset));
    seg_file->Write((char *)&len, offset + ofst, sizeof(len));

    seg_file->WriteStr(value, sizeof(char) * len);
  }
}

int Table::AddField(const string &name, DataType ftype, bool is_index) {
  if (attr_idx_map_.find(name) != attr_idx_map_.end()) {
    LOG(ERROR) << "Duplicate field " << name;
    return -1;
  }
  if (name == "_id") {
    key_idx_ = field_num_;
    id_type_ = ftype == DataType::STRING ? 0 : 1;
  }
  if (ftype == DataType::STRING) {
    ++string_field_num_;
  }
  idx_attr_offset_.push_back(item_length_);
  item_length_ += FTypeSize(ftype);
  attrs_.push_back(ftype);
  idx_attr_map_.insert(std::pair<int, string>(field_num_, name));
  attr_idx_map_.insert(std::pair<string, int>(name, field_num_));
  attr_type_map_.insert(std::pair<string, DataType>(name, ftype));
  attr_is_index_map_.insert(std::pair<string, bool>(name, is_index));
  ++field_num_;
  return 0;
}

int Table::GetDocIDByKey(std::string &key, int &docid) {
#ifdef USE_BTREE
  BtDb *bt = bt_open(cache_mgr_, main_mgr_);
  int ret = bt_findkey(bt, reinterpret_cast<unsigned char *>(&key), sizeof(key),
                       (unsigned char *)&docid, sizeof(int));
  bt_close(bt);

  if (ret >= 0) {
    return 0;
  }
#else
  if (id_type_ == 0) {
    int64_t k = utils::StringToInt64(key);
    if (item_to_docid_.find(k, docid)) {
      return 0;
    }
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    if (item_to_docid_.find(key_long, docid)) {
      return 0;
    }
  }

#endif
  return -1;
}

int Table::Add(const std::string &key, const std::vector<struct Field> &fields,
               int docid) {
  if (fields.size() != attr_idx_map_.size()) {
    LOG(ERROR) << "Field num [" << fields.size() << "] not equal to ["
               << attr_idx_map_.size() << "]";
    return -2;
  }
  if (key.size() == 0) {
    LOG(ERROR) << "Add item error : _id is null!";
    return -3;
  }

#ifdef USE_BTREE
  BtDb *bt = bt_open(cache_mgr_, main_mgr_);

  BTERR bterr = bt_insertkey(
      bt->main, reinterpret_cast<unsigned char *>(&key.data()), sizeof(key), 0,
      static_cast<void *>(&docid), sizeof(int), Unique);
  if (bterr) {
    LOG(ERROR) << "Error " << bt->mgr->err;
  }
  bt_close(bt);
#else
  if (id_type_ == 0) {
    int64_t k = utils::StringToInt64(key);
    item_to_docid_.insert(k, docid);
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    item_to_docid_.insert(key_long, docid);
  }
#endif

  for (size_t i = 0; i < fields.size(); ++i) {
    const auto &field_value = fields[i];
    const string &name = field_value.name;

    SetFieldValue(docid, name, i, field_value.value.c_str(),
                  field_value.value.size());
  }

  TableData *seg_file = main_file_[docid / DOCNUM_PER_SEGMENT];
  seg_file->SetSize(seg_file->Size() + 1);
  // if (PutToDB(docid)) {
  //   LOG(ERROR) << "Put to rocksdb error, docid [" << docid << "]";
  //   return -2;
  // }

  Compress();

  if (docid % 10000 == 0) {
    if (id_type_ == 0) {
      LOG(INFO) << "Add item _id [" << key << "], num [" << docid << "]";
    } else {
      long key_long = -1;
      memcpy(&key_long, key.data(), sizeof(key_long));
      LOG(INFO) << "Add item _id [" << key_long << "], num [" << docid << "]";
    }
  }
  last_docid_ = docid;
  return 0;
}

int Table::BatchAdd(int start_id, int batch_size, int docid,
                    std::vector<Doc> &doc_vec, BatchResult &result) {
#ifdef PERFORMANCE_TESTING
  double start = utils::getmillisecs();
#endif

#pragma omp parallel for
  for (size_t i = 0; i < batch_size; ++i) {
    int id = docid + i;
    Doc &doc = doc_vec[start_id + i];

    std::string &key = doc.Key();
    if (key.size() == 0) {
      std::string msg = "Add item error : _id is null!";
      result.SetResult(i, -1, msg);
      LOG(ERROR) << msg;
      continue;
    }

    if (id_type_ == 0) {
      int64_t k = utils::StringToInt64(key);
      item_to_docid_.insert(k, id);
    } else {
      long key_long = -1;
      memcpy(&key_long, key.data(), sizeof(key_long));

      item_to_docid_.insert(key_long, id);
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    int id = docid + i;
    Doc &doc = doc_vec[start_id + i];
    std::vector<Field> &fields = doc.TableFields();
    for (size_t j = 0; j < attr_idx_map_.size(); ++j) {
      const auto &field_value = fields[j];
      const string &name = field_value.name;

      SetFieldValue(id, name, j, field_value.value.c_str(),
                    field_value.value.size());
    }
    if (id % 10000 == 0) {
      std::string &key = doc_vec[i].Key();
      if (id_type_ == 0) {
        LOG(INFO) << "Add item _id [" << key << "], num [" << id << "]";
      } else {
        long key_long = -1;
        memcpy(&key_long, key.data(), sizeof(key_long));
        LOG(INFO) << "Add item _id [" << key_long << "], num [" << id << "]";
      }
    }
    TableData *seg_file = main_file_[id / DOCNUM_PER_SEGMENT];
    seg_file->SetSize(seg_file->Size() + 1);
  }

  // if (BatchPutToDB(docid, batch_size)) {
  //   LOG(ERROR) << "put to rocksdb error, docid=" << docid;
  //   return -2;
  // }

  Compress();
#ifdef PERFORMANCE_TESTING
  double end = utils::getmillisecs();
  if (docid % 10000 == 0) {
    LOG(INFO) << "table cost [" << end - start << "]ms";
  }
#endif
  last_docid_ = docid + batch_size;
  return 0;
}

int Table::Update(const std::vector<Field> &fields, int docid) {
  if (fields.size() == 0) return 0;

  for (size_t i = 0; i < fields.size(); ++i) {
    const struct Field &field_value = fields[i];
    const string &name = field_value.name;
    const auto &it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Cannot find field name [" << name << "]";
      continue;
    }

    int field_id = it->second;

    if (field_value.datatype == DataType::STRING) {
      int offset = idx_attr_offset_[field_id];

      int seg_pos;
      size_t in_seg_pos;
      int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos, false);
      if (ret != 0) {
        return ret;
      }
      offset += in_seg_pos * item_length_;
      TableData *seg_file = main_file_[seg_pos];
      char *base = seg_file->Base();

      str_offset_t str_offset = 0;
      memcpy(&str_offset, base + offset, sizeof(str_offset));
      str_len_t len;
      memcpy(&len, base + offset + sizeof(str_offset), sizeof(len));

      size_t value_len = field_value.value.size();
      if (len >= value_len) {
        seg_file->Write((char *)&value_len, offset + sizeof(str_offset),
                        sizeof(len));
        seg_file->WriteStr(field_value.value.data(), str_offset, value_len);
      } else {
        len = value_len;
        int ofst = sizeof(str_offset);
        str_offset = seg_file->StrOffset();
        seg_file->Write((char *)&str_offset, offset, sizeof(str_offset));
        seg_file->Write((char *)&len, offset + ofst, sizeof(len));
        seg_file->WriteStr(field_value.value.data(), sizeof(char) * len);
      }
    } else {
      SetFieldValue(docid, name, field_id, field_value.value.data(),
                    field_value.value.size());
    }
  }

  // if (PutToDB(docid)) {
  //   LOG(ERROR) << "update to rocksdb error, docid=" << docid;
  //   return -2;
  // }

  Compress();

  return 0;
}

int Table::Delete(std::string &key) {
  if (id_type_ == 0) {
    int64_t k = utils::StringToInt64(key);
    item_to_docid_.erase(k);
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    item_to_docid_.erase(key_long);
  }
  return 0;
}

int Table::GetRawDoc(int docid, vector<char> &raw_doc) {
  int len = item_length_;
  raw_doc.resize(len, 0);
  int seg_pos;
  size_t in_seg_pos;
  int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
  if (ret != 0) {
    return ret;
  }
  size_t offset = in_seg_pos * item_length_;
  TableData *seg_file = main_file_[seg_pos];
  char *base = seg_file->Base();
  memcpy((void *)raw_doc.data(), base + offset, item_length_);
  DecompressStr decompress_str;

  for (int i = 0; i < (int)idx_attr_offset_.size(); i++) {
    if (attrs_[i] != DataType::STRING) continue;

    char *field = base + offset + idx_attr_offset_[i];
    str_len_t str_len = 0;
    memcpy((void *)&str_len, field + sizeof(str_offset_t), sizeof(str_len));
    if (str_len == 0) continue;

    raw_doc.resize(len + str_len, 0);
    str_offset_t str_offset = 0;
    memcpy((void *)&str_offset, field, sizeof(str_offset));
    std::string str;
    int ret = seg_file->GetStr(str_offset, str_len, str, decompress_str);
    if (ret != 0) {
      LOG(ERROR) << "Get str error [" << docid << "] len [" << (int)str_len
                 << "]";
    }
    memcpy((void *)(raw_doc.data() + len), str.c_str(), str_len);
    len += str_len;
  }
  return 0;
}

int Table::GetSegPos(IN int32_t docid, IN int32_t field_id, OUT int &seg_pos,
                     OUT size_t &in_seg_pos, bool bRead) {
  seg_pos = docid / DOCNUM_PER_SEGMENT;
  if (seg_pos >= seg_num_) {
    if (bRead) {
      LOG(ERROR) << "Pos [" << seg_pos << "] out of bound [" << seg_num_ << "]";
      return -1;
    }
    int ret = Extend();
    if (ret != 0) {
      LOG(ERROR) << "docid [" << docid << "], main_file [" << seg_pos
                 << "] is NULL";
      return -1;
    }
  }
  in_seg_pos = docid % DOCNUM_PER_SEGMENT;
  return 0;
}

int Table::Extend() {
  main_file_[seg_num_] = new TableData(item_length_);
  main_file_[seg_num_]->Init(seg_num_, root_path_, string_field_num_);
  ++seg_num_;
  return 0;
}

void Table::Compress() {
  if (b_compress_) {
    if (seg_num_ < 2) return;

    for (int i = compressed_num_; i < seg_num_ - 1; ++i) {
      main_file_[i]->Compress();
    }

    compressed_num_ = seg_num_ - 1;
  }
}

long Table::GetMemoryBytes() {
  long total_mem_bytes = 0;
  for (int i = 0; i < seg_num_; ++i) {
    total_mem_bytes += main_file_[i]->GetMemoryBytes();
  }
  return total_mem_bytes;
}

int Table::GetDocInfo(std::string &id, Doc &doc,
                      DecompressStr &decompress_str) {
  int doc_id = 0;
  int ret = GetDocIDByKey(id, doc_id);
  if (ret < 0) {
    return ret;
  }
  return GetDocInfo(doc_id, doc, decompress_str);
}

int Table::GetDocInfo(const int docid, Doc &doc,
                      DecompressStr &decompress_str) {
  if (docid > last_docid_) {
    LOG(ERROR) << "doc [" << docid << "] in front of [" << last_docid_ << "]";
    return -1;
  }
  int i = 0;
  std::vector<struct Field> &table_fields = doc.TableFields();
  table_fields.resize(attr_type_map_.size());

  for (const auto &it : attr_type_map_) {
    const string &attr = it.first;
    GetFieldInfo(docid, attr, table_fields[i], decompress_str);
    ++i;
  }
  return 0;
}

void Table::GetFieldInfo(const int docid, const string &field_name,
                         struct Field &field, DecompressStr &decompress_str) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return;
  }

  DataType type = it->second;

  std::string source;
  field.name = field_name;
  field.source = source;
  field.datatype = type;

  if (type == DataType::STRING) {
    GetFieldString(docid, field_name, field.value, decompress_str);
  } else {
    int value_len = FTypeSize(type);

    std::string str_value;
    if (type == DataType::INT) {
      int value = 0;
      GetField<int>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == DataType::LONG) {
      long value = 0;
      GetField<long>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == DataType::FLOAT) {
      float value = 0;
      GetField<float>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == DataType::DOUBLE) {
      double value = 0;
      GetField<double>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    }
    field.value = std::move(str_value);
  }
}

int Table::GetFieldString(int docid, const std::string &field,
                          std::string &value, DecompressStr &decompress_str) {
  const auto &iter = attr_idx_map_.find(field);
  if (iter == attr_idx_map_.end()) {
    LOG(ERROR) << "docid " << docid << " field " << field;
    return -1;
  }
  int idx = iter->second;
  return GetFieldString(docid, idx, value, decompress_str);
}

int Table::GetFieldString(int docid, int field_id, std::string &value,
                          DecompressStr &decompress_str) {
  size_t offset = idx_attr_offset_[field_id];
  str_offset_t str_offset = 0;

  int seg_pos;
  size_t in_seg_pos;
  int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
  if (ret != 0) {
    return ret;
  }
  offset += in_seg_pos * item_length_;
  TableData *seg_file = main_file_[seg_pos];
  if (seg_pos == decompress_str.SegID()) {
    decompress_str.SetHit(true);
  } else {
    decompress_str.SetHit(false);
  }
  decompress_str.SetSegID(seg_pos);

  char *base = seg_file->Base();

  memcpy(&str_offset, base + offset, sizeof(str_offset));

  str_len_t len;
  memcpy(&len, base + offset + sizeof(str_offset), sizeof(len));
  ret = seg_file->GetStr(str_offset, len, value, decompress_str);
  if (ret != 0) {
    decompress_str.SetHit(false);
  }
  return ret;
}

int Table::GetFieldRawValue(int docid, int field_id, std::string &value) {
  if ((docid < 0) or (field_id < 0 || field_id >= field_num_)) return -1;

  DataType data_type = attrs_[field_id];
  if (data_type != DataType::STRING) {
    size_t offset = idx_attr_offset_[field_id];
    int data_len = FTypeSize(data_type);
    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
    if (ret != 0) {
      return ret;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    char *base = seg_file->Base();
    value = std::string(base + offset, data_len);
  } else {
    DecompressStr decompress_str;
    GetFieldString(docid, field_id, value, decompress_str);
  }
  return 0;
}

int Table::GetFieldType(const std::string &field_name, DataType &type) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return -1;
  }
  type = it->second;
  return 0;
}

int Table::GetAttrType(std::map<std::string, DataType> &attr_type_map) {
  for (const auto attr_type : attr_type_map_) {
    attr_type_map.insert(attr_type);
  }
  return 0;
}

int Table::GetAttrIsIndex(std::map<std::string, bool> &attr_is_index_map) {
  for (const auto attr_is_index : attr_is_index_map_) {
    attr_is_index_map.insert(attr_is_index);
  }
  return 0;
}

int Table::GetAttrIdx(const std::string &field) const {
  const auto &iter = attr_idx_map_.find(field.c_str());
  return (iter != attr_idx_map_.end()) ? iter->second : -1;
}

int Table::AddRawDoc(int docid, const char *raw_doc, int doc_size) {
  int seg_pos;
  size_t in_seg_pos;
  int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
  if (ret != 0) {
    return ret;
  }
  size_t offset = in_seg_pos * item_length_;
  TableData *seg_file = main_file_[seg_pos];
  char *base = seg_file->Base();
  uint64_t str_offset = seg_file->StrOffset();

  memcpy((void *)(base + offset), raw_doc, item_length_);
  raw_doc += item_length_;
  seg_file->WriteStr(raw_doc, doc_size - item_length_);

  for (size_t field_id = 0; field_id < idx_attr_offset_.size(); ++field_id) {
    if (attrs_[field_id] != DataType::STRING) continue;

    int field_offset = idx_attr_offset_[field_id];
    char *field = base + offset + field_offset;  // TODO base is read only
    memcpy((void *)field, (void *)&str_offset, sizeof(str_offset));
    str_len_t field_len = 0;
    memcpy((void *)&field_len, (field + sizeof(str_offset)), sizeof(field_len));
    str_offset += field_len;
  }
  return 0;
}

}  // namespace table
}  // namespace tig_gamma
