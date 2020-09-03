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
#ifdef WITH_ROCKSDB
using namespace rocksdb;
#endif

namespace table {

const static string kTableDumpedNum = "table_dumped_num";

Table::Table(const string &root_path) {
  item_length_ = 0;
  field_num_ = 0;
  key_idx_ = -1;
  db_path_ = root_path + "/table";
  seg_num_ = 0;

  // TODO : there is a failure.
  // if (!item_to_docid_.reserve(max_doc_size)) {
  //   LOG(ERROR) << "item_to_docid reserve failed, max_doc_size [" <<
  //   max_doc_size
  //              << "]";
  // }

  table_created_ = false;
  LOG(INFO) << "Table created success!";
}

Table::~Table() {
#ifdef WITH_ROCKSDB
  if (db_) {
    delete db_;
    db_ = nullptr;
  }
#endif

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
  LOG(INFO) << "Table deleted.";
}

int Table::Load(const std::vector<string> &folders, int &doc_num) {
  doc_num = 0;
#ifdef WITH_ROCKSDB
  string value;
  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), kTableDumpedNum, &value);
  if (!s.ok()) {
    LOG(INFO) << "the key=" << kTableDumpedNum << " isn't in db, skip loading";
    doc_num = 0;
    return 0;
  }
  doc_num = std::stoi(value, 0, 10);
  if (doc_num < 0) {
    LOG(ERROR) << "invalid doc num of db, value=" << value;
    return -1;
  }
  LOG(INFO) << "begin to load table, doc num=" << doc_num;
  rocksdb::Iterator *it = db_->NewIterator(rocksdb::ReadOptions());
  string start_key;
  ToRowKey(0, start_key);
  it->Seek(Slice(start_key));
  for (int c = 0; c < doc_num; c++, it->Next()) {
    if (!it->Valid()) {
      LOG(ERROR) << "rocksdb iterator error, count=" << c;
      delete it;
      return 2;
    }
    Slice value = it->value();
    const char *data = value.data_;

    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(c, 0, seg_pos, in_seg_pos);
    if (ret != 0) {
      return ret;
    }
    size_t offset = in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    char *base = seg_file->Base();
    str_offset_t str_offset = seg_file->StrOffset();

    memcpy((void *)(base + offset), data, item_length_);
    data += item_length_;
    seg_file->WriteStr(data, value.size_ - item_length_);

    for (size_t field_id = 0; field_id < idx_attr_offset_.size(); ++field_id) {
      if (attrs_[field_id] != tig_gamma::DataType::STRING) continue;

      int field_offset = idx_attr_offset_[field_id];
      char *field = base + offset + field_offset;
      memcpy((void *)field, (void *)&str_offset, sizeof(str_offset));
      str_len_t field_len = 0;
      memcpy((void *)&field_len, (field + sizeof(str_offset)),
             sizeof(field_len));
      str_offset += field_len;
    }
  }
  delete it;

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
      item_to_docid_str_.insert(std::move(key), i);
    } else {
      long key = -1;
      GetField<long>(i, idx, key);
      item_to_docid_.insert(key, i);
    }
  }

  LOG(INFO) << "Table load successed! doc num=" << doc_num;
#else
  LOG(ERROR) << "rocksdb is need when compiling for table's loading";
#endif
  return 0;
}

int Table::CreateTable(tig_gamma::TableInfo &table) {
  if (table_created_) {
    return -10;
  }
  name_ = table.Name();
  std::vector<struct tig_gamma::FieldInfo> &fields = table.Fields();

  size_t fields_num = fields.size();
  for (size_t i = 0; i < fields_num; ++i) {
    const string name = fields[i].name;
    tig_gamma::DataType ftype = fields[i].data_type;
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

#ifdef WITH_ROCKSDB
  // open DB
  Options options;
  options.IncreaseParallelism();
  // options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  Status s = DB::Open(options, db_path_, &db_);
  if (!s.ok()) {
    LOG(ERROR) << "open rocks db error: " << s.ToString();
    return -1;
  }
#endif

#ifdef USE_BTREE
  uint mainleafxtra = 0;
  uint maxleaves = 1000000;
  uint poolsize = 500;
  uint leafxtra = 0;
  uint mainpool = 500;
  uint mainbits = 16;
  uint bits = 16;

  if (!utils::isFolderExist(db_path_.c_str())) {
    mkdir(db_path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  string cache_file = db_path_ + string("/cache_") + ".dis";
  string main_file = db_path_ + string("/main_") + ".dis";

  remove(cache_file.c_str());
  remove(main_file.c_str());

  cache_mgr_ =
      bt_mgr(const_cast<char *>(cache_file.c_str()), bits, leafxtra, poolsize);
  cache_mgr_->maxleaves = maxleaves;
  main_mgr_ = bt_mgr(const_cast<char *>(main_file.c_str()), mainbits,
                     mainleafxtra, mainpool);
  main_mgr_->maxleaves = maxleaves;
#endif

  table_created_ = true;
  LOG(INFO) << "Create table " << name_ << " success!";
  return 0;
}

int Table::FTypeSize(tig_gamma::DataType fType) {
  int length = 0;
  if (fType == tig_gamma::DataType::INT) {
    length = sizeof(int32_t);
  } else if (fType == tig_gamma::DataType::LONG) {
    length = sizeof(int64_t);
  } else if (fType == tig_gamma::DataType::FLOAT) {
    length = sizeof(float);
  } else if (fType == tig_gamma::DataType::DOUBLE) {
    length = sizeof(double);
  } else if (fType == tig_gamma::DataType::STRING) {
    length = sizeof(str_offset_t) + sizeof(str_len_t);
  }
  return length;
}

void Table::SetFieldValue(int docid, const std::string &field, int field_id,
                          const char *value, str_len_t len) {
  size_t offset = idx_attr_offset_[field_id];
  tig_gamma::DataType attr = attrs_[field_id];

  if (attr != tig_gamma::DataType::STRING) {
    int type_size = FTypeSize(attr);

    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
    if (ret != 0) {
      return;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    char *base = seg_file->Base();
    memcpy(base + offset, value, type_size);
  } else {
    size_t ofst = sizeof(str_offset_t);
    int seg_pos;
    size_t in_seg_pos;
    int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
    if (ret != 0) {
      return;
    }
    offset += in_seg_pos * item_length_;
    TableData *seg_file = main_file_[seg_pos];
    char *base = seg_file->Base();
    str_offset_t str_offset = seg_file->StrOffset();
    memcpy(base + offset, &str_offset, sizeof(str_offset));
    memcpy(base + offset + ofst, &len, sizeof(len));

    seg_file->WriteStr(value, sizeof(char) * len);
  }
}

int Table::AddField(const string &name, tig_gamma::DataType ftype,
                    bool is_index) {
  if (attr_idx_map_.find(name) != attr_idx_map_.end()) {
    LOG(ERROR) << "Duplicate field " << name;
    return -1;
  }
  if (name == "_id") {
    key_idx_ = field_num_;
    id_type_ = ftype == tig_gamma::DataType::STRING ? 0 : 1;
  }
  idx_attr_offset_.push_back(item_length_);
  item_length_ += FTypeSize(ftype);
  attrs_.push_back(ftype);
  idx_attr_map_.insert(std::pair<int, string>(field_num_, name));
  attr_idx_map_.insert(std::pair<string, int>(name, field_num_));
  attr_type_map_.insert(std::pair<string, tig_gamma::DataType>(name, ftype));
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

int Table::Add(const std::vector<struct tig_gamma::Field> &fields, int docid) {
  if (fields.size() != attr_idx_map_.size()) {
    LOG(ERROR) << "Field num [" << fields.size() << "] not equal to ["
               << attr_idx_map_.size() << "]";
    return -2;
  }
  std::vector<tig_gamma::Field> fields_reorder(fields.size());
  std::string key;
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto &field_value = fields[i];
    const string &name = field_value.name;

    auto it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Unknown field " << name;
      continue;
    }
    int field_idx = it->second;
    fields_reorder[field_idx] = field_value;
    if (name == "_id") {
      key = field_value.value;
    }
  }
#ifdef DEBUG__
  printDoc(doc);
#endif
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

  for (size_t i = 0; i < fields_reorder.size(); ++i) {
    const auto &field_value = fields_reorder[i];
    const string &name = field_value.name;

    SetFieldValue(docid, name.c_str(), i, field_value.value.c_str(),
                  field_value.value.size());
  }

  if (PutToDB(docid)) {
    LOG(ERROR) << "put to rocksdb error, docid=" << docid;
    return -2;
  }
  if (docid % 10000 == 0) {
    if (id_type_ == 0) {
      LOG(INFO) << "Add item _id [" << key << "], num [" << docid << "]";
    } else {
      long key_long = -1;
      memcpy(&key_long, key.data(), sizeof(key_long));
      LOG(INFO) << "Add item _id [" << key_long << "], num [" << docid << "]";
    }
  }
  return 0;
}

int Table::Update(const std::vector<tig_gamma::Field> &fields, int docid) {
  if (fields.size() == 0) return 0;

  for (size_t i = 0; i < fields.size(); ++i) {
    const struct tig_gamma::Field &field_value = fields[i];
    const string &name = field_value.name;
    const auto &it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Cannot find field name [" << name << "]";
      continue;
    }

    int field_id = it->second;

    if (field_value.datatype == tig_gamma::DataType::STRING) {
      int offset = idx_attr_offset_[field_id];

      int seg_pos;
      size_t in_seg_pos;
      int ret = GetSegPos(docid, 0, seg_pos, in_seg_pos);
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
        memcpy(base + offset + sizeof(str_offset), &(value_len), sizeof(len));

        seg_file->WriteStr(field_value.value.data(), str_offset, value_len);
      } else {
        len = value_len;
        int ofst = sizeof(str_offset);
        str_offset = seg_file->StrOffset();
        memcpy(base + offset, &str_offset, sizeof(str_offset));
        memcpy(base + offset + ofst, &len, sizeof(len));
        seg_file->WriteStr(field_value.value.data(), sizeof(char) * len);
      }
    } else {
      SetFieldValue(docid, name.c_str(), field_id, field_value.value.data(),
                    field_value.value.size());
    }
  }

  if (PutToDB(docid)) {
    LOG(ERROR) << "update to rocksdb error, docid=" << docid;
    return -2;
  }

  return 0;
}

int Table::Delete(std::string &key) {
  if (id_type_ == 0) {
    item_to_docid_str_.erase(key);
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    item_to_docid_.erase(key_long);
  }
  return 0;
}

void Table::ToRowKey(int id, string &key) const {
  char data[11];
  snprintf(data, 11, "%010d", id);
  key.assign(data, 10);
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
    if (attrs_[i] != tig_gamma::DataType::STRING) continue;

    char *field = base + offset + idx_attr_offset_[i];
    str_len_t str_len = 0;
    memcpy((void *)&str_len, field + sizeof(str_offset_t), sizeof(str_len));
    if (str_len == 0) continue;

    raw_doc.resize(len + str_len, 0);
    str_offset_t str_offset = 0;
    memcpy((void *)&str_offset, field, sizeof(str_offset));
    std::string str;
    seg_file->GetStr(str_offset, str_len, str, decompress_str);
    memcpy((void *)(raw_doc.data() + len), str.c_str(), str_len);
    len += str_len;
  }
  return 0;
}

int Table::GetSegPos(IN int32_t docid, IN int32_t field_id, OUT int &seg_pos,
                     OUT size_t &in_seg_pos) {
  seg_pos = docid / DOCNUM_PER_SEGMENT;
  if (seg_pos >= seg_num_) {
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

int Table::PutToDB(int docid) {
#ifdef WITH_ROCKSDB
  vector<char> doc;
  GetRawDoc(docid, doc);
  string key;
  ToRowKey(docid, key);
  Status s = db_->Put(WriteOptions(), Slice(key),
                      Slice((char *)doc.data(), doc.size()));
  if (!s.ok()) {
    LOG(ERROR) << "rocksdb put error:" << s.ToString() << ", key=" << key;
    return -2;
  }
#endif
  return 0;
}

int Table::Extend() {
  if (seg_num_ >= 1) {
    main_file_[seg_num_ - 1]->Compress();
  }

  main_file_[seg_num_] = new TableData(item_length_);
  main_file_[seg_num_]->Init();
  ++seg_num_;
  return 0;
}

int Table::Dump(const string &path, int start_docid, int end_docid) {
#ifdef WITH_ROCKSDB
  string value;
  ToRowKey(end_docid + 1, value);
  Status s = db_->Put(WriteOptions(), Slice(kTableDumpedNum), Slice(value));
  if (!s.ok()) {
    LOG(ERROR) << "rocksdb put error:" << s.ToString()
               << ", key=" << kTableDumpedNum << ", value=" << value;
    return -2;
  }
#else
  LOG(ERROR) << "rocksdb is need when compiling for table's dumping";
#endif
  return 0;
}

long Table::GetMemoryBytes() { return 0; }

int Table::GetDocInfo(std::string &id, tig_gamma::Doc &doc,
                      DecompressStr &decompress_str) {
  int doc_id = 0;
  int ret = GetDocIDByKey(id, doc_id);
  if (ret < 0) {
    return ret;
  }
  return GetDocInfo(doc_id, doc, decompress_str);
}

int Table::GetDocInfo(const int docid, tig_gamma::Doc &doc,
                      DecompressStr &decompress_str) {
  int i = 0;
  for (const auto &it : attr_type_map_) {
    const string &attr = it.first;
    struct tig_gamma::Field field;
    GetFieldInfo(docid, attr, field, decompress_str);
    doc.AddField(std::move(field));
    ++i;
  }
  return 0;
}

void Table::GetFieldInfo(const int docid, const string &field_name,
                         struct tig_gamma::Field &field,
                         DecompressStr &decompress_str) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return;
  }

  tig_gamma::DataType type = it->second;

  std::string source;
  field.name = field_name;
  field.source = source;
  field.datatype = type;

  if (type == tig_gamma::DataType::STRING) {
    GetFieldString(docid, field_name, field.value, decompress_str);
  } else {
    int value_len = FTypeSize(type);

    std::string str_value;
    if (type == tig_gamma::DataType::INT) {
      int value = 0;
      GetField<int>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == tig_gamma::DataType::LONG) {
      long value = 0;
      GetField<long>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == tig_gamma::DataType::FLOAT) {
      float value = 0;
      GetField<float>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    } else if (type == tig_gamma::DataType::DOUBLE) {
      double value = 0;
      GetField<double>(docid, field_name, value);
      str_value = std::string(reinterpret_cast<char *>(&value), value_len);
    }
    field.value = str_value;
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
  seg_file->GetStr(str_offset, len, value, decompress_str);
  return 0;
}

int Table::GetFieldRawValue(int docid, int field_id, std::string &value) {
  if ((docid < 0) or (field_id < 0 || field_id >= field_num_)) return -1;

  tig_gamma::DataType data_type = attrs_[field_id];
  if (data_type != tig_gamma::DataType::STRING) {
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

int Table::GetFieldType(const std::string &field_name,
                        tig_gamma::DataType &type) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return -1;
  }
  type = it->second;
  return 0;
}

int Table::GetAttrType(
    std::map<std::string, tig_gamma::DataType> &attr_type_map) {
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

}  // namespace table
