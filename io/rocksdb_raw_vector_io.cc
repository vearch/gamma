#ifdef WITH_ROCKSDB

#include "rocksdb_raw_vector_io.h"

namespace tig_gamma {

using std::string;
using namespace rocksdb;

int RocksDBRawVectorIO::Load(int vec_num) {
  if (vec_num == 0) return 0;
  string key, value;
  raw_vector->ToRowKey(vec_num - 1, key);
  Status s = raw_vector->db_->Get(ReadOptions(), Slice(key), &value);
  if (!s.ok()) {
    LOG(ERROR) << "load vectors, get error:" << s.ToString() << ", expected key=" << key;
    return INTERNAL_ERR;
  }
  raw_vector->MetaInfo()->size_ = vec_num;
  LOG(INFO) << "rocksdb load success! vec_num=" << vec_num;
  return 0;
}

}  // namespace tig_gamma

#endif // WITH_ROCKSDB
