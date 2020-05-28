/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "profile.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <fstream>
#include <string>

#include "utils.h"

using std::move;
using std::string;
using std::vector;
#ifdef WITH_ROCKSDB
using namespace rocksdb;
#endif

namespace tig_gamma {

const static string kProfileDumpedNum = "profile_dumped_num";

Profile::Profile(const int max_doc_size, const string &root_path) {
  item_length_ = 0;
  field_num_ = 0;
  key_idx_ = -1;
  mem_ = nullptr;
  str_mem_ = nullptr;
  max_profile_size_ = max_doc_size;
  max_str_size_ = max_profile_size_ * 128;
  str_offset_ = 0;
  db_path_ = root_path + "/profile";

  // TODO : there is a failure.
  // if (!item_to_docid_.reserve(max_doc_size)) {
  //   LOG(ERROR) << "item_to_docid reserve failed, max_doc_size [" <<
  //   max_doc_size
  //              << "]";
  // }

  table_created_ = false;
  LOG(INFO) << "Profile created success!";
}

Profile::~Profile() {
  if (mem_ != nullptr) {
    delete[] mem_;
  }

  if (str_mem_ != nullptr) {
    delete[] str_mem_;
  }

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
  LOG(INFO) << "Profile deleted.";
}

int Profile::Load(const std::vector<string> &folders, int &doc_num) {
  doc_num = 0;
#ifdef WITH_ROCKSDB
  string value;
  rocksdb::Status s =
      db_->Get(rocksdb::ReadOptions(), kProfileDumpedNum, &value);
  if (!s.ok()) {
    LOG(INFO) << "the key=" << kProfileDumpedNum
              << " isn't in db, skip loading";
    doc_num = 0;
    return 0;
  }
  doc_num = std::stoi(value, 0, 10);
  if (doc_num < 0) {
    LOG(ERROR) << "invalid doc num of db, value=" << value;
    return -1;
  }
  LOG(INFO) << "begin to load profile, doc num=" << doc_num;
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
    memcpy((void *)(mem_ + (long)c * item_length_), data, item_length_);
    data += item_length_;
    memcpy(str_mem_ + str_offset_, data, value.size_ - item_length_);
    for (int field_id = 0; field_id < (int)idx_attr_offset_.size();
         field_id++) {
      if (attrs_[field_id] != STRING) continue;
      int field_offset = idx_attr_offset_[field_id];
      char *field = mem_ + (long)c * item_length_ + field_offset;
      memcpy((void *)field, (void *)&str_offset_, sizeof(uint64_t));
      uint16_t field_len = 0;
      memcpy((void *)&field_len, (field + sizeof(uint64_t)), sizeof(field_len));
      str_offset_ += field_len;
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
    long key = -1;
    GetField<long>(i, idx, key);
#ifdef USE_BTREE
    BtDb *bt = bt_open(cache_mgr_, main_mgr_);
    BTERR bterr = bt_insertkey(
        bt->main, reinterpret_cast<unsigned char *>(&key), sizeof(key), 0,
        static_cast<void *>(&i), sizeof(int), Unique);
    if (bterr) {
      LOG(ERROR) << "Error " << bt->mgr->err;
    }
    bt_close(bt);
#else
    item_to_docid_.insert(key, i);
#endif
  }

  LOG(INFO) << "Profile load successed! doc num=" << doc_num;
#else
  LOG(ERROR) << "rocksdb is need when compiling for profile's loading";
#endif
  return 0;
}

int Profile::CreateTable(const Table *table) {
  if (table_created_) {
    return -10;
  }
  name_ = std::string(table->name->value, table->name->len);
  for (int i = 0; i < table->fields_num; ++i) {
    const string name =
        string(table->fields[i]->name->value, table->fields[i]->name->len);
    enum DataType ftype = table->fields[i]->data_type;
    int is_index = table->fields[i]->is_index;
    LOG(INFO) << "Add field name [" << name << "], type [" << ftype
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

  id_type_ = table->id_type;

  if (mem_) {
    delete[] mem_;
  }
  if (str_mem_) {
    delete[] str_mem_;
  }

  mem_ = new char[max_profile_size_ * item_length_];
  str_mem_ = new char[max_str_size_];

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

int Profile::FTypeSize(enum DataType fType) {
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
    length = sizeof(uint64_t) + sizeof(uint16_t);
  }
  return length;
}

void Profile::SetFieldValue(int docid, const std::string &field,
                            const char *value, uint16_t len) {
  const auto &iter = attr_idx_map_.find(field);
  if (iter == attr_idx_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field << "]";
    return;
  }
  int idx = iter->second;
  size_t offset = (uint64_t)docid * item_length_ + idx_attr_offset_[idx];
  enum DataType attr = attrs_[idx];

  if (attr != DataType::STRING) {
    int type_size = FTypeSize(attr);
    memcpy(mem_ + offset, value, type_size);
  } else {
    int ofst = sizeof(uint64_t);
    if ((str_offset_ + len) >= max_str_size_) {
      LOG(ERROR) << "Str memory reached max size [" << max_str_size_ << "]";
      return;
    }
    memcpy(mem_ + offset, &str_offset_, sizeof(uint64_t));
    memcpy(mem_ + offset + ofst, &len, sizeof(uint16_t));
    memcpy(str_mem_ + str_offset_, value, sizeof(char) * len);
    str_offset_ += len;
  }
}

int Profile::AddField(const string &name, enum DataType ftype, int is_index) {
  if (attr_idx_map_.find(name) != attr_idx_map_.end()) {
    LOG(ERROR) << "Duplicate field " << name;
    return -1;
  }
  if (name == "_id") {
    key_idx_ = field_num_;
  }
  idx_attr_offset_.push_back(item_length_);
  item_length_ += FTypeSize(ftype);
  attrs_.push_back(ftype);
  idx_attr_map_.insert(std::pair<int, string>(field_num_, name));
  attr_idx_map_.insert(std::pair<string, int>(name, field_num_));
  attr_type_map_.insert(std::pair<string, enum DataType>(name, ftype));
  attr_is_index_map_.insert(std::pair<string, int>(name, is_index));
  ++field_num_;
  return 0;
}

int Profile::GetDocIDByKey(std::string &key, int &doc_id) {
#ifdef USE_BTREE
  BtDb *bt = bt_open(cache_mgr_, main_mgr_);
  int ret = bt_findkey(bt, reinterpret_cast<unsigned char *>(&key), sizeof(key),
                       (unsigned char *)&doc_id, sizeof(int));
  bt_close(bt);

  if (ret >= 0) {
    return 0;
  }
#else
  if (id_type_ == 0) {
    if (item_to_docid_str_.find(key, doc_id)) {
      return 0;
    }
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    if (item_to_docid_.find(key_long, doc_id)) {
      return 0;
    }
  }

#endif
  return -1;
}

int Profile::Add(const std::vector<Field *> &fields, int doc_id,
                 bool is_existed) {
  if (doc_id >= static_cast<int>(max_profile_size_)) {
    LOG(ERROR) << "Doc num reached upper limit [" << max_profile_size_ << "]";
    return -1;
  }

  if (fields.size() != attr_idx_map_.size()) {
    LOG(ERROR) << "Field num [" << fields.size() << "] not equal to ["
               << attr_idx_map_.size() << "]";
    return -1;
  }
  std::vector<Field *> fields_reorder(fields.size());
  std::string key;
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto field_value = fields[i];
    const string &name =
        std::string(field_value->name->value, field_value->name->len);

    auto it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Unknown field " << name;
      continue;
    }
    int field_idx = it->second;
    fields_reorder[field_idx] = field_value;
    if (name == "_id") {
      key = std::string(field_value->value->value, field_value->value->len);
    }
  }
#ifdef DEBUG__
  printDoc(doc);
#endif
  if (key.size() == 0) {
    LOG(ERROR) << "Add item error : _id is null!";
    return -1;
  }

#ifdef USE_BTREE
  BtDb *bt = bt_open(cache_mgr_, main_mgr_);

  BTERR bterr = bt_insertkey(bt->main, reinterpret_cast<unsigned char *>(&key.data()),
                             sizeof(key), 0, static_cast<void *>(&doc_id),
                             sizeof(int), Unique);
  if (bterr) {
    LOG(ERROR) << "Error " << bt->mgr->err;
  }
  bt_close(bt);
#else
if (id_type_ == 0) {
    item_to_docid_str_.insert(key, doc_id);
  } else {
    long key_long = -1;
    memcpy(&key_long, key.data(), sizeof(key_long));

    item_to_docid_.insert(key_long, doc_id);
  }

#endif

  for (size_t i = 0; i < fields_reorder.size(); ++i) {
    const auto field_value = fields_reorder[i];
    const string &name =
        std::string(field_value->name->value, field_value->name->len);

    auto it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Cannot find field name [" << name << "]";
      continue;
    }
    SetFieldValue(doc_id, name.c_str(), field_value->value->value,
                  field_value->value->len);
  }

  if (doc_id % 10000 == 0) {
    LOG(INFO) << "Add item _id [" << key << "], num [" << doc_id << "]"
              << ", is_existed=" << is_existed;
  }
  return 0;
}

int Profile::Update(const std::vector<Field *> &fields, int doc_id) {
  if (fields.size() == 0) return 0;

  for (size_t i = 0; i < fields.size(); ++i) {
    const auto field_value = fields[i];
    const string &name =
        string(field_value->name->value, field_value->name->len);
    const auto &it = attr_idx_map_.find(name);
    if (it == attr_idx_map_.end()) {
      LOG(ERROR) << "Cannot find field name [" << name << "]";
      continue;
    }

    int field_id = it->second;

    if (field_value->data_type == STRING) {
      size_t offset =
          (uint64_t)doc_id * item_length_ + idx_attr_offset_[field_id];
      size_t str_offset = 0;
      memcpy(&str_offset, mem_ + offset, sizeof(size_t));
      unsigned short len;
      memcpy(&len, mem_ + offset + sizeof(size_t), sizeof(unsigned short));

      if (len >= field_value->value->len) {
        memcpy(mem_ + offset + sizeof(size_t), &(field_value->value->len),
               sizeof(unsigned short));
        memcpy(str_mem_ + str_offset, field_value->value->value,
               field_value->value->len);
      } else {
        len = field_value->value->len;
        int ofst = sizeof(uint64_t);
        if ((str_offset_ + len) >= max_str_size_) {
          LOG(ERROR) << "Str memory reached max size [" << max_str_size_ << "]";
          return -1;
        }
        memcpy(mem_ + offset, &str_offset_, sizeof(uint64_t));
        memcpy(mem_ + offset + ofst, &len, sizeof(uint16_t));
        memcpy(str_mem_ + str_offset_, field_value->value->value,
               sizeof(char) * len);
        str_offset_ += len;
      }
    } else {
      SetFieldValue(doc_id, name.c_str(), field_value->value->value,
                    field_value->value->len);
    }
  }

  if (PutToDB(doc_id)) {
    LOG(ERROR) << "update to rocksdb error, docid=" << doc_id;
    return -2;
  }

  return 0;
}

void Profile::ToRowKey(int id, string &key) const {
  char data[11];
  snprintf(data, 11, "%010d", id);
  key.assign(data, 10);
}

int Profile::GetRawDoc(int docid, vector<char> &raw_doc) {
  int len = item_length_;
  raw_doc.resize(len, 0);
  memcpy((void *)raw_doc.data(), mem_ + (long)docid * item_length_,
         item_length_);
  for (int i = 0; i < (int)idx_attr_offset_.size(); i++) {
    if (attrs_[i] != STRING) continue;
    char *field = mem_ + (long)docid * item_length_ + idx_attr_offset_[i];
    uint16_t str_len = 0;
    memcpy((void *)&str_len, field + sizeof(uint64_t), sizeof(str_len));
    if (str_len == 0) continue;
    raw_doc.resize(len + str_len, 0);
    uint64_t str_offset = 0;
    memcpy((void *)&str_offset, field, sizeof(uint64_t));
    memcpy((void *)(raw_doc.data() + len), str_mem_ + str_offset, str_len);
    len += str_len;
  }
  return 0;
}

int Profile::PutToDB(int docid) {
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

int Profile::Dump(const string &path, int start_docid, int end_docid) {
#ifdef WITH_ROCKSDB
  for (int docid = start_docid; docid <= end_docid; docid++) {
    if (PutToDB(docid)) {
      LOG(ERROR) << "put to rocksdb error, docid=" << docid;
      return -2;
    }
  }
  string value;
  ToRowKey(end_docid + 1, value);
  Status s = db_->Put(WriteOptions(), Slice(kProfileDumpedNum), Slice(value));
  if (!s.ok()) {
    LOG(ERROR) << "rocksdb put error:" << s.ToString()
               << ", key=" << kProfileDumpedNum << ", value=" << value;
    return -2;
  }
#else
  LOG(ERROR) << "rocksdb is need when compiling for profile's dumping";
#endif
  return 0;
}

long Profile::GetMemoryBytes() {
  return max_profile_size_ * item_length_ + max_str_size_;
}

int Profile::GetDocInfo(const int docid, Doc *&doc) {
  if (doc == nullptr) {
    doc = static_cast<Doc *>(malloc(sizeof(Doc)));
    doc->fields_num = attr_type_map_.size();
    doc->fields =
        static_cast<Field **>(malloc(doc->fields_num * sizeof(Field *)));
    memset(doc->fields, 0, doc->fields_num * sizeof(Field *));
  }

  int i = 0;
  for (const auto &it : attr_type_map_) {
    const string &attr = it.first;
    doc->fields[i] = GetFieldInfo(docid, attr);
    ++i;
  }

  return 0;
}

Field *Profile::GetFieldInfo(const int docid, const string &field_name) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return nullptr;
  }

  enum DataType type = it->second;
  Field *field = static_cast<Field *>(malloc(sizeof(Field)));
  memset(field, 0, sizeof(Field));
  field->name = StringToByteArray(field_name);
  field->value = static_cast<ByteArray *>(malloc(sizeof(ByteArray)));

  if (type != DataType::STRING) {
    field->value->len = FTypeSize(type);
    field->value->value = static_cast<char *>(malloc(field->value->len));
  }

  if (type == DataType::INT) {
    int value = 0;
    GetField<int>(docid, field_name, value);
    memcpy(field->value->value, &value, field->value->len);
  } else if (type == DataType::LONG) {
    long value = 0;
    GetField<long>(docid, field_name, value);
    memcpy(field->value->value, &value, field->value->len);
  } else if (type == DataType::FLOAT) {
    float value = 0;
    GetField<float>(docid, field_name, value);
    memcpy(field->value->value, &value, field->value->len);
  } else if (type == DataType::DOUBLE) {
    double value = 0;
    GetField<double>(docid, field_name, value);
    memcpy(field->value->value, &value, field->value->len);
  } else if (type == DataType::STRING) {
    char *value;
    field->value->len = GetFieldString(docid, field_name, &value);
    field->value->value = static_cast<char *>(malloc(field->value->len));
    memcpy(field->value->value, value, field->value->len);
  }
  field->data_type = type;
  return field;
}

int Profile::GetDocInfo(ByteArray *key, Doc *&doc) {
  int doc_id = 0;
  std::string key_str = std::string(key->value, key->len);
  int ret = GetDocIDByKey(key_str, doc_id);
  if (ret < 0) {
    return ret;
  }
  return GetDocInfo(doc_id, doc);
}

int Profile::GetFieldString(int docid, const std::string &field,
                            char **value) const {
  const auto &iter = attr_idx_map_.find(field);
  if (iter == attr_idx_map_.end()) {
    LOG(ERROR) << "docid " << docid << " field " << field;
    return -1;
  }
  int idx = iter->second;
  return GetFieldString(docid, idx, value);
}

int Profile::GetFieldString(int docid, int field_id, char **value) const {
  size_t offset = (uint64_t)docid * item_length_ + idx_attr_offset_[field_id];
  size_t str_offset = 0;
  memcpy(&str_offset, mem_ + offset, sizeof(size_t));
  unsigned short len;
  memcpy(&len, mem_ + offset + sizeof(size_t), sizeof(unsigned short));
  offset += sizeof(unsigned short);
  *value = str_mem_ + str_offset;
  return len;
}

int Profile::GetFieldRawValue(int docid, int field_id, unsigned char **value,
                              int &data_len) {
  if ((docid < 0) or (field_id < 0 || field_id >= field_num_)) return -1;

  enum DataType data_type = attrs_[field_id];
  if (data_type != DataType::STRING) {
    size_t offset = (uint64_t)docid * item_length_ + idx_attr_offset_[field_id];
    data_len = FTypeSize(data_type);
    *value = reinterpret_cast<unsigned char *>(mem_ + offset);
  } else {
    data_len =
        GetFieldString(docid, field_id, reinterpret_cast<char **>(value));
  }
  return 0;
}

int Profile::GetFieldType(const std::string &field_name, enum DataType &type) {
  const auto &it = attr_type_map_.find(field_name);
  if (it == attr_type_map_.end()) {
    LOG(ERROR) << "Cannot find field [" << field_name << "]";
    return -1;
  }
  type = it->second;
  return 0;
}

int Profile::GetAttrType(std::map<std::string, enum DataType> &attr_type_map) {
  for (const auto attr_type : attr_type_map_) {
    attr_type_map.insert(attr_type);
  }
  return 0;
}

int Profile::GetAttrIsIndex(std::map<std::string, int> &attr_is_index_map) {
  for (const auto attr_is_index : attr_is_index_map_) {
    attr_is_index_map.insert(attr_is_index);
  }
  return 0;
}

int Profile::GetAttrIdx(const std::string &field) const {
  const auto &iter = attr_idx_map_.find(field.c_str());
  return (iter != attr_idx_map_.end()) ? iter->second : -1;
}

}  // namespace tig_gamma
