/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "vector_manager.h"

#include "raw_vector_factory.h"
#include "utils.h"

namespace tig_gamma {

static bool InnerProductCmp(const VectorDoc *a, const VectorDoc *b) {
  return a->score > b->score;
}

static bool L2Cmp(const VectorDoc *a, const VectorDoc *b) {
  return a->score < b->score;
}

VectorManager::VectorManager(const VectorStorageType &store_type,
                             const char *docids_bitmap,
                             const std::string &root_path)
    : default_store_type_(store_type),
      docids_bitmap_(docids_bitmap),
      root_path_(root_path) {
  table_created_ = false;
}

VectorManager::~VectorManager() { Close(); }

int VectorManager::CreateVectorTable(TableInfo &table) {
  if (table_created_) return -1;

  std::map<std::string, int> vec_dups;

  std::vector<struct VectorInfo> &vectors_infos = table.VectorInfos();

  for (struct VectorInfo &vectors_info : vectors_infos) {
    std::string &name = vectors_info.name;
    auto it = vec_dups.find(name);
    if (it == vec_dups.end()) {
      vec_dups[name] = 1;
    } else {
      ++vec_dups[name];
    }
  }

  for (size_t i = 0; i < vectors_infos.size(); i++) {
    struct VectorInfo &vector_info = vectors_infos[i];
    std::string &vec_name = vector_info.name;
    int dimension = vector_info.dimension;

    std::string &store_type_str = vector_info.store_type;

    VectorStorageType store_type = default_store_type_;
    if (store_type_str != "") {
      if (!strcasecmp("MemoryOnly", store_type_str.c_str())) {
        store_type = VectorStorageType::MemoryOnly;
      } else if (!strcasecmp("Mmap", store_type_str.c_str())) {
        store_type = VectorStorageType::Mmap;
#ifdef WITH_ROCKSDB
      } else if (!strcasecmp("RocksDB", store_type_str.c_str())) {
        store_type = VectorStorageType::RocksDB;
#endif  // WITH_ROCKSDB
      } else {
        LOG(WARNING) << "NO support for store type " << store_type_str;
        return -1;
      }
    }

    std::string &store_param = vector_info.store_param;

    std::string &retrieval_type_str = table.RetrievalType();
    std::string &retrieval_param = table.RetrievalParam();

    // TODO : this should configed by schema
    VectorValueType value_type = VectorValueType::FLOAT;
    if (retrieval_type_str == "BINARYIVF") {
      value_type = VectorValueType::BINARY;
      dimension /= 8;
    }

    string vec_root_path = root_path_ + "/vectors";
    if (utils::make_dir(vec_root_path.c_str())) {
      LOG(ERROR) << "make directory error, path=" << vec_root_path;
      return -2;
    }
    VectorMetaInfo *meta_info =
        new VectorMetaInfo(vec_name, dimension, value_type);

    RawVector *vec = RawVectorFactory::Create(
        meta_info, store_type, vec_root_path, store_param, docids_bitmap_);
    if (vec == nullptr) {
      LOG(ERROR) << "create raw vector error";
      return -1;
    }
    bool has_source = vector_info.has_source;
    bool multi_vids = vec_dups[vec_name] > 1 ? true : false;
    int ret = vec->Init(has_source, multi_vids);
    if (ret != 0) {
      LOG(ERROR) << "Raw vector " << vec_name << " init error, code [" << ret
                 << "]!";
      delete vec;
      return -1;
    }

    StartFlushingIfNeed(vec);
    raw_vectors_[vec_name] = vec;

    if (vector_info.is_index == false) {
      LOG(INFO) << vec_name << " need not to indexed!";
      continue;
    }

    LOG(INFO) << "Create index model [" << retrieval_type_str << "]";
    RetrievalModel *retrieval_model = dynamic_cast<RetrievalModel *>(
        reflector().GetNewModel(retrieval_type_str));
    if (retrieval_model == nullptr) {
      LOG(ERROR) << "Cannot get model=" << retrieval_type_str
                 << ", vec_name=" << vec_name;
      delete vec;
      return -1;
    }
    retrieval_model->vector_ = vec;
    if (retrieval_model->Init(retrieval_param) != 0) {
      LOG(ERROR) << "gamma index init " << vec_name << " error!";
      delete vec;
      delete retrieval_model;
      return -1;
    }
    vector_indexes_[vec_name] = retrieval_model;
  }
  table_created_ = true;
  return 0;
}

int VectorManager::AddToStore(int docid, std::vector<struct Field> &fields) {
  for (size_t i = 0; i < fields.size(); i++) {
    std::string &name = fields[i].name;
    if (raw_vectors_.find(name) == raw_vectors_.end()) {
      LOG(ERROR) << "Cannot find raw vector [" << name << "]";
      continue;
    }
    raw_vectors_[name]->Add(docid, fields[i]);
  }
  return 0;
}

int VectorManager::Update(int docid, std::vector<Field> &fields) {
  for (size_t i = 0; i < fields.size(); i++) {
    string &name = fields[i].name;
    auto it = raw_vectors_.find(name);
    if (it == raw_vectors_.end()) {
      continue;
    }
    RawVector *raw_vector = it->second;
    if ((size_t)raw_vector->MetaInfo()->Dimension() !=
        fields[i].value.size() / sizeof(float)) {
      LOG(ERROR) << "invalid field value len=" << fields[i].value.size()
                 << ", dimension=" << raw_vector->MetaInfo()->Dimension();
      return -1;
    }

    return raw_vector->Update(docid, fields[i]);
  }

  return 0;
}

int VectorManager::Delete(int docid) {
  for (const auto &iter : vector_indexes_) {
    std::vector<int64_t> vids;
    RawVector *vector = dynamic_cast<RawVector *>(iter.second->vector_);
    vector->VidMgr()->DocID2VID(docid, vids);
    if (0 != iter.second->Delete(vids)) {
      LOG(ERROR) << "delete index from" << iter.first
                 << " failed! docid=" << docid;
      return -1;
    }
  }
  return 0;
}

int VectorManager::Indexing() {
  int ret = 0;
  for (const auto &iter : vector_indexes_) {
    if (0 != iter.second->Indexing()) {
      ret = -1;
      LOG(ERROR) << "vector table " << iter.first << " indexing failed!";
    }
  }
  return ret;
}

int VectorManager::AddRTVecsToIndex() {
  int ret = 0;
  for (const auto &iter : vector_indexes_) {
    RawVector *raw_vec = dynamic_cast<RawVector *>(iter.second->vector_);
    int total_stored_vecs = raw_vec->MetaInfo()->Size();
    int indexed_vec_count = raw_vec->IndexedVectorNum();

    if (indexed_vec_count > total_stored_vecs) {
      LOG(ERROR) << "internal error : indexed_vec_count=" << indexed_vec_count
                 << " should not greater than total_stored_vecs="
                 << total_stored_vecs;
      ret = -1;
    } else if (indexed_vec_count == total_stored_vecs) {
#ifdef DEBUG
      LOG(INFO) << "no extra vectors existed for indexing";
#endif
    } else {
      int MAX_NUM_PER_INDEX = 1000;
      int index_count =
          (total_stored_vecs - indexed_vec_count) / MAX_NUM_PER_INDEX + 1;

      for (int i = 0; i < index_count; i++) {
        int start_docid = raw_vec->IndexedVectorNum();
        size_t count_per_index =
            (i == (index_count - 1) ? total_stored_vecs - start_docid
                                    : MAX_NUM_PER_INDEX);
        if (count_per_index == 0) break;

        std::vector<int> lens;
        ScopeVectors vector_head;
        raw_vec->GetVectorHeader(start_docid, count_per_index, vector_head,
                                 lens);
        const uint8_t *add_vec = nullptr;
        utils::ScopeDeleter1<uint8_t> del_vec;

        if (lens.size() == 1) {
          add_vec = vector_head.Get(0);
        } else {
          int raw_d = raw_vec->MetaInfo()->Dimension();
          if (raw_vec->MetaInfo()->DataType() == VectorValueType::BINARY) {
            raw_d /= 8;
            add_vec = new uint8_t[raw_d * count_per_index];
          } else {
            add_vec = new uint8_t[raw_d * count_per_index * sizeof(float)];
          }
          del_vec.set(add_vec);
          size_t offset = 0;
          for (size_t i = 0; i < vector_head.Size(); ++i) {
            memcpy((void *)(add_vec + offset), (void *)vector_head.Get(i),
                   sizeof(float) * raw_d * lens[i]);

            if (raw_vec->MetaInfo()->DataType() == VectorValueType::BINARY) {
              offset += raw_d * lens[i];
            } else {
              offset += sizeof(float) * raw_d * lens[i];
            }
          }
        }
        if (!iter.second->Add(count_per_index, add_vec)) {
          LOG(ERROR) << "add index from docid " << start_docid << " error!";
          ret = -2;
        } else {
          int indexed_count = raw_vec->IndexedVectorNum() + count_per_index;
          raw_vec->SetIndexedVectorNum(indexed_count);
        }
      }
    }
    std::vector<int64_t> vids;
    int vid;
    while (raw_vec->UpdatedVids()->try_dequeue(vid)) {
      if (bitmap::test(raw_vec->Bitmap(), raw_vec->VidMgr()->VID2DocID(vid)))
        continue;
      vids.push_back(vid);
      if (vids.size() >= 20000) break;
    }
    if (vids.size() == 0) continue;
    ScopeVectors scope_vecs;
    raw_vec->Gets(vids, scope_vecs);
    if (iter.second->Update(vids, scope_vecs.Get())) {
      LOG(ERROR) << "update index error!";
      ret = -3;
    }
  }
  return ret;
}

namespace {

int parse_index_search_result(int n, int k, VectorResult &result,
                              RetrievalModel *index) {
  RawVector *raw_vec = dynamic_cast<RawVector *>(index->vector_);
  if (raw_vec == nullptr) {
    LOG(ERROR) << "Cannot get raw vector";
    return -1;
  }
  for (int i = 0; i < n; i++) {
    int pos = 0;

    std::map<int, int> docid2count;
    for (int j = 0; j < k; j++) {
      int64_t *docid = result.docids + i * k + j;
      if (docid[0] == -1) continue;
      int vector_id = (int)docid[0];
      int real_docid = raw_vec->VidMgr()->VID2DocID(vector_id);

      if (docid2count.find(real_docid) == docid2count.end()) {
        int real_pos = i * k + pos;
        result.docids[real_pos] = real_docid;
        int ret = raw_vec->GetSource(vector_id, result.sources[real_pos],
                                     result.source_lens[real_pos]);
        if (ret != 0) {
          result.sources[real_pos] = nullptr;
          result.source_lens[real_pos] = 0;
        }
        result.dists[real_pos] = result.dists[i * k + j];

        pos++;
        docid2count[real_docid] = 1;
      }
    }

    if (pos > 0) {
      result.idx[i] = 0;  // init start id of seeking
    }

    for (; pos < k; pos++) {
      result.docids[i * k + pos] = -1;
      result.dists[i * k + pos] = -1;
    }
  }
  return 0;
}

}  // namespace

int VectorManager::Search(GammaQuery &query, GammaResult *results) {
  int ret = 0, n = 0;

  size_t vec_num = query.vec_query.size();
  VectorResult all_vector_results[vec_num];

  query.condition->sort_by_docid = vec_num > 1 ? true : false;
  std::string vec_names[vec_num];
  for (size_t i = 0; i < vec_num; i++) {
    struct VectorQuery &vec_query = query.vec_query[i];

    std::string &name = vec_query.name;
    vec_names[i] = name;

    std::map<std::string, RetrievalModel *>::iterator iter =
        vector_indexes_.find(name);
    if (iter == vector_indexes_.end()) {
      LOG(ERROR) << "Query name " << name
                 << " not exist in created vector table";
      return -1;
    }

    RetrievalModel *index = iter->second;
    RawVector *raw_vec = dynamic_cast<RawVector *>(iter->second->vector_);
    int d = raw_vec->MetaInfo()->Dimension();
    if (raw_vec->MetaInfo()->DataType() == VectorValueType::BINARY) {
      n = vec_query.value.size() / d;
    } else {
      n = vec_query.value.size() / (raw_vec->MetaInfo()->DataSize() * d);
    }

    if (n <= 0) {
      LOG(ERROR) << "Search n shouldn't less than 0!";
      return -1;
    }

    if (!all_vector_results[i].init(n, query.condition->topn)) {
      LOG(ERROR) << "Query name " << name << "init vector result error";
      return -2;
    }

    query.condition->Init(vec_query.min_score, vec_query.max_score,
                          docids_bitmap_, raw_vec);
    query.condition->retrieval_params_ =
        index->Parse(query.condition->retrieval_parameters);
    query.condition->metric_type =
        query.condition->retrieval_params_->GetDistanceComputeType();

    const uint8_t *x =
        reinterpret_cast<const uint8_t *>(vec_query.value.c_str());
    int ret_vec = index->Search(query.condition, n, x, query.condition->topn,
                                all_vector_results[i].dists,
                                all_vector_results[i].docids);

    if (ret_vec != 0) {
      ret = ret_vec;
      LOG(ERROR) << "faild search of query " << name;
      return -3;
    } else {
      parse_index_search_result(n, query.condition->topn, all_vector_results[i],
                                index);

      if (query.condition->sort_by_docid) {
        all_vector_results[i].sort_by_docid();
      }
    }
#ifdef PERFORMANCE_TESTING
    std::string msg;
    msg += "search " + std::to_string(i);
    query.condition->GetPerfTool().Perf(msg);
#endif
  }

  if (query.condition->sort_by_docid) {
    for (int i = 0; i < n; i++) {
      int start_docid = 0, common_idx = 0;
      size_t common_docid_count = 0;
      double score = 0;
      bool has_common_docid = true;
      if (!results[i].init(query.condition->topn, vec_names, vec_num)) {
        LOG(ERROR) << "init gamma result(sort by docid) error, topn="
                   << query.condition->topn << ", vector number=" << vec_num;
        return -4;
      }
      while (start_docid < INT_MAX) {
        for (size_t j = 0; j < vec_num; j++) {
          float vec_dist = 0;
          char *source = nullptr;
          int source_len = 0;
          int cur_docid = all_vector_results[j].seek(i, start_docid, vec_dist,
                                                     source, source_len);
          if (cur_docid == start_docid) {
            common_docid_count++;
            double field_score = query.vec_query[j].has_boost == 1
                                     ? (vec_dist * query.vec_query[j].boost)
                                     : vec_dist;
            score += field_score;
            results[i].docs[common_idx]->fields[j].score = field_score;
            results[i].docs[common_idx]->fields[j].source = source;
            results[i].docs[common_idx]->fields[j].source_len = source_len;
            if (common_docid_count == vec_num) {
              results[i].docs[common_idx]->docid = start_docid;
              results[i].docs[common_idx++]->score = score;
              results[i].total = all_vector_results[j].total[i] > 0
                                     ? all_vector_results[j].total[i]
                                     : results[i].total;

              start_docid++;
              common_docid_count = 0;
              score = 0;
            }
          } else if (cur_docid > start_docid) {
            common_docid_count = 0;
            start_docid = cur_docid;
            score = 0;
          } else {
            has_common_docid = false;
            break;
          }
        }
        if (!has_common_docid) break;
      }
      results[i].results_count = common_idx;
      if (query.condition->multi_vector_rank) {
        switch (query.condition->metric_type) {
          case DistanceComputeType::INNER_PRODUCT:
            std::sort(results[i].docs, results[i].docs + common_idx,
                      InnerProductCmp);
            break;
          case DistanceComputeType::L2:
            std::sort(results[i].docs, results[i].docs + common_idx, L2Cmp);
            break;
          default:
            LOG(ERROR) << "invalid metric_type="
                       << (int)query.condition->metric_type;
        }
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      // double score = 0;
      if (!results[i].init(query.condition->topn, vec_names, vec_num)) {
        LOG(ERROR) << "init gamma result error, topn=" << query.condition->topn
                   << ", vector number=" << vec_num;
        return -5;
      }
      results[i].total = all_vector_results[0].total[i] > 0
                             ? all_vector_results[0].total[i]
                             : results[i].total;
      int pos = 0, topn = all_vector_results[0].topn;
      for (int j = 0; j < topn; j++) {
        int real_pos = i * topn + j;
        if (all_vector_results[0].docids[real_pos] == -1) continue;
        results[i].docs[pos]->docid = all_vector_results[0].docids[real_pos];

        results[i].docs[pos]->fields[0].source =
            all_vector_results[0].sources[real_pos];
        results[i].docs[pos]->fields[0].source_len =
            all_vector_results[0].source_lens[real_pos];

        double score = all_vector_results[0].dists[real_pos];

        score = query.vec_query[0].has_boost == 1
                    ? (score * query.vec_query[0].boost)
                    : score;

        results[i].docs[pos]->fields[0].score = score;
        results[i].docs[pos]->score = score;
        pos++;
      }
      results[i].results_count = pos;
    }
  }

#ifdef PERFORMANCE_TESTING
  query.condition->GetPerfTool().Perf("merge result");
#endif
  return ret;
}

int VectorManager::GetVector(
    const std::vector<std::pair<string, int>> &fields_ids,
    std::vector<string> &vec, bool is_bytearray) {
  for (const auto &pair : fields_ids) {
    const string &field = pair.first;
    const int id = pair.second;
    auto iter = raw_vectors_.find(field);
    if (iter == raw_vectors_.end()) {
      continue;
    }
    RawVector *raw_vec = iter->second;
    if (raw_vec == nullptr) {
      LOG(ERROR) << "raw_vec is null!";
      return -1;
    }
    int vid = raw_vec->VidMgr()->GetFirstVID(id);

    char *source = nullptr;
    int len = -1;
    int ret = raw_vec->GetSource(vid, source, len);

    if (ret != 0 || len < 0) {
      LOG(ERROR) << "Get source failed!";
      return -1;
    }

    ScopeVector scope_vec;
    raw_vec->GetVector(vid, scope_vec);
    const float *feature = (const float *)(scope_vec.Get());
    std::string str_vec;
    if (is_bytearray) {
      int d = raw_vec->MetaInfo()->Dimension();
      int d_byte = d * raw_vec->MetaInfo()->DataSize();

      char feat_source[sizeof(d) + d_byte + len];

      memcpy((void *)feat_source, &d_byte, sizeof(int));
      int cur = sizeof(d_byte);

      memcpy((void *)(feat_source + cur), feature, d_byte);
      cur += d_byte;

      memcpy((void *)(feat_source + cur), source, len);

      str_vec =
          string((char *)feat_source, sizeof(unsigned int) + d_byte + len);
    } else {
      VectorValueType data_type = raw_vec->MetaInfo()->DataType();
      if (data_type == VectorValueType::FLOAT) {
        const float *feature_float = reinterpret_cast<const float *>(feature);
        for (int i = 0; i < raw_vec->MetaInfo()->Dimension(); ++i) {
          str_vec += std::to_string(feature_float[i]) + ",";
        }
      } else if (data_type == VectorValueType::BINARY) {
        for (int i = 0; i < raw_vec->MetaInfo()->Dimension(); ++i) {
          str_vec += std::to_string(feature[i]) + ",";
        }
      }
      str_vec.pop_back();
    }
    vec.emplace_back(std::move(str_vec));
  }
  return 0;
}

void VectorManager::GetTotalMemBytes(long &index_total_mem_bytes,
                                     long &vector_total_mem_bytes) {
  for (const auto &iter : vector_indexes_) {
    index_total_mem_bytes += iter.second->GetTotalMemBytes();
  }

  for (const auto &iter : raw_vectors_) {
    vector_total_mem_bytes += iter.second->GetTotalMemBytes();
  }
}

int VectorManager::Dump(const string &path, int dump_docid, int max_docid) {
  for (const auto &iter : vector_indexes_) {
    const string &vec_name = iter.first;
    RetrievalModel *index = iter.second;
    int ret = index->Dump(path);
    if (ret != 0) {
      LOG(ERROR) << "vector " << vec_name << " dump gamma index failed!";
      return -1;
    }
    LOG(INFO) << "vector " << vec_name << " dump gamma index success!";
  }

  for (const auto &iter : raw_vectors_) {
    const string &vec_name = iter.first;
    RawVector *raw_vector = dynamic_cast<RawVector *>(iter.second);
    int ret = raw_vector->Dump(path, dump_docid, max_docid);
    if (ret != 0) {
      LOG(ERROR) << "vector " << vec_name << " dump failed!";
      return -1;
    }
    LOG(INFO) << "vector " << vec_name << " dump success!";
  }

  return 0;
}

int VectorManager::Load(const std::vector<std::string> &index_dirs,
                        int doc_num) {
  for (const auto &iter : raw_vectors_) {
    if (0 != iter.second->Load(index_dirs, doc_num)) {
      LOG(ERROR) << "vector [" << iter.first << "] load failed!";
      return -1;
    }
    LOG(INFO) << "vector [" << iter.first << "] load success!";
  }

  if (index_dirs.size() > 0) {
    for (const auto &iter : vector_indexes_) {
      if (iter.second->Load(index_dirs[0]) < 0) {
        LOG(ERROR) << "vector [" << iter.first << "] load gamma index failed!";
        return -1;
      } else {
        LOG(INFO) << "vector [" << iter.first << "] load gamma index success!";
      }
    }
  }

  return 0;
}

RetrievalModel *VectorManager::GetVectorIndex(std::string &name) const {
  const auto &it = vector_indexes_.find(name);
  if (it == vector_indexes_.end()) {
    return nullptr;
  }
  return it->second;
}

void VectorManager::Close() {
  for (const auto &iter : raw_vectors_) {
    if (iter.second != nullptr) {
      StopFlushingIfNeed(iter.second);
      delete iter.second;
    }
  }
  raw_vectors_.clear();
  LOG(INFO) << "Raw vector cleared.";

  for (const auto &iter : vector_indexes_) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
  vector_indexes_.clear();
  LOG(INFO) << "Vector indexes cleared.";

  LOG(INFO) << "VectorManager closed.";
}
}  // namespace tig_gamma
