/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "api_data/gamma_table.h"
#include "utils.h"

namespace tig_gamma {

class TableIO {
 public:
  TableIO(std::string &file_path);

  ~TableIO();

  int Write(TableInfo &table);

  void WriteFieldInfos(TableInfo &table);

  void WriteVectorInfos(TableInfo &table);

  void WriteRetrievalType(TableInfo &table);

  void WriteRetrievalParam(TableInfo &table);

  int Read(std::string &name, TableInfo &table);

  void ReadFieldInfos(TableInfo &table);

  void ReadVectorInfos(TableInfo &table);

  void ReadRetrievalType(TableInfo &table);

  void ReadRetrievalParam(TableInfo &table);

  utils::FileIO *fio;
};

}  // namespace tig_gamma