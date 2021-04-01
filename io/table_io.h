#pragma once

#include <string>

#include "async_flush.h"
#include "rocksdb_wrapper.h"
#include "table.h"
#include "utils.h"

namespace tig_gamma {

class TableIO : public AsyncFlusher {
 public:
  table::Table *table;

  TableIO(table::Table *table_) : AsyncFlusher("table"), table(table_) {}
  virtual ~TableIO() {}

  int Init();
  int Dump(int start_docid, int end_docid);
  int Load(int &doc_num);

  int FlushOnce() override;
};

}  // namespace tig_gamma
