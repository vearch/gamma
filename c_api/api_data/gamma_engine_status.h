/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "api_data/gamma_raw_data.h"
#include "engine_status_generated.h"

namespace tig_gamma {

class EngineStatus : public RawData {
 public:
  EngineStatus();

  virtual int Serialize(char **out, int *out_len);

  virtual void Deserialize(const char *data, int len);

  int IndexStatus();

  void SetIndexStatus(int index_status);

  long TableMem();

  void SetTableMem(long table_mem_bytes);

  long IndexMem();

  void SetIndexMem(long index_mem_bytes);

  long VectorMem();

  void SetVectorMem(long vector_mem_bytes);

  long FieldRangeMem();

  void SetFieldRangeMem(long field_range_mem_bytes);

  long BitmapMem();

  void SetBitmapMem(long bitmap_mem_bytes);

  int DocNum();

  void SetDocNum(int doc_num);

 private:
  gamma_api::EngineStatus *engine_status_;

  int index_status_;
  long table_mem_bytes_;
  long index_mem_bytes_;
  long vector_mem_bytes_;
  long field_range_mem_bytes_;
  long bitmap_mem_bytes_;

  int doc_num_;
};

}  // namespace tig_gamma