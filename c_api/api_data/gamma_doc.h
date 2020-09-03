/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "doc_generated.h"
#include "gamma_raw_data.h"
#include "gamma_table.h"

namespace tig_gamma {

struct Field {
  std::string name;
  std::string value;
  std::string source;
  DataType datatype;

  Field() {}

  Field(const Field &other) {
    name = other.name;
    value = other.value;
    source = other.source;
    datatype = other.datatype;
  }

  Field &operator=(const Field &other) {
    name = other.name;
    value = other.value;
    source = other.source;
    datatype = other.datatype;
    return *this;
  }

  Field(Field &&other) {
    name = std::move(other.name);
    value = std::move(other.value);
    source = std::move(other.source);
    datatype = other.datatype;
  }

  Field &operator=(Field &&other) {
    name = std::move(other.name);
    value = std::move(other.value);
    source = std::move(other.source);
    datatype = other.datatype;
    return *this;
  }
};

class Doc : public RawData {
 public:
  Doc() { doc_ = nullptr; }

  virtual int Serialize(char **out, int *out_len);

  virtual void Deserialize(const char *data, int len);

  void AddField(const struct Field &field);

  void AddField(struct Field &&field);

  std::vector<struct Field> &Fields();

 private:
  gamma_api::Doc *doc_;

  std::vector<struct Field> fields_;
};

}  // namespace tig_gamma
