#include <string>
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

namespace tig_gamma {

struct RocksDBWrapper {
  rocksdb::DB *db_;
  rocksdb::BlockBasedTableOptions table_options_;

  RocksDBWrapper();
  ~RocksDBWrapper();
  int Open(std::string db_path, size_t block_cache_size = 0);
  int Put(int key, const char *v, size_t len);
  void ToRowKey(int key, std::string &key_str);
};

}  // namespace tig_gamma
