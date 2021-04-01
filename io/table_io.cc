#include "table_io.h"

#include "io_common.h"

namespace tig_gamma {

int TableIO::Init() {
  // do nothing
  return 0;
}

int TableIO::Dump(int start_docid, int end_docid) {
  // int ret = table->Sync();
  return 0;
}

int TableIO::Load(int &doc_num) {
  table->Load(doc_num);
  Reset(doc_num);
  LOG(INFO) << "Table load successed! doc num=" << doc_num;
  return 0;
}

int TableIO::FlushOnce() { return 0; }

}  // namespace tig_gamma
