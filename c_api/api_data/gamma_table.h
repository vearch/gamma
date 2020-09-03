/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "config_generated.h"
#include "doc_generated.h"
#include "gamma_raw_data.h"
#include "response_generated.h"
#include "table_generated.h"

namespace tig_gamma {

enum class DataType : std::uint16_t {
  INT = 0,
  LONG,
  FLOAT,
  DOUBLE,
  STRING,
  VECTOR
};

struct VectorInfo {
  std::string name;
  DataType data_type;
  bool is_index;
  int dimension;
  std::string model_id;
  std::string store_type;
  std::string store_param;
  bool has_source;
};

struct FieldInfo {
  std::string name;
  DataType data_type;
  bool is_index;
};

class TableInfo : public RawData {
 public:
  TableInfo() { table_ = nullptr; }

  virtual int Serialize(char **out, int *out_len);
  virtual void Deserialize(const char *data, int len);

  std::string &Name();

  void SetName(std::string &name);

  std::vector<struct FieldInfo> &Fields();

  void AddField(struct FieldInfo &field);

  std::vector<struct VectorInfo> &VectorInfos();

  void AddVectorInfo(struct VectorInfo &vector_info);

  int IndexingSize();

  void SetIndexingSize(int indexing_size);

  std::string &RetrievalType();

  void SetRetrievalType(std::string &retrieval_type);

  std::string &RetrievalParam();

  void SetRetrievalParam(std::string &retrieval_param);

  int Read(const std::string &path);

  int Write(const std::string &path);

 private:
  gamma_api::Table *table_;

  std::string name_;
  std::vector<struct FieldInfo> fields_;
  std::vector<struct VectorInfo> vectors_infos_;

  int indexing_size_;
  std::string retrieval_type_;
  std::string retrieval_param_;
};

}  // namespace tig_gamma
