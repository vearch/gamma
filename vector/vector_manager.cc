/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "vector_manager.h"

#include "gamma_index_factory.h"
#include "raw_vector_factory.h"
#include "utils.h"

namespace tig_gamma {

static bool InnerProductCmp(const VectorDoc *a, const VectorDoc *b) {
  return a->score > b->score;
}

static bool L2Cmp(const VectorDoc *a, const VectorDoc *b) {
  return a->score < b->score;
}

VectorManager::VectorManager(const RetrievalModel &model,
                             const VectorStorageType &store_type,
                             const char *docids_bitmap, int max_doc_size,
                             const std::string &root_path,
                             GammaCounters *counters)
    : default_model_(model),
      default_store_type_(store_type),
      docids_bitmap_(docids_bitmap),
      max_doc_size_(max_doc_size),
      root_path_(root_path),
      gamma_counters_(counters) {
  table_created_ = false;
  retrieval_param_ = nullptr;
}

VectorManager::~VectorManager() { Close(); }

int VectorManager::CreateVectorTable(VectorInfo **vectors_info, int vectors_num,
                                     std::string &retrieval_type,
                                     std::string &retrieval_param) {
  if (table_created_) return -1;

  retrieval_param_ = new RetrievalParams();
  if (retrieval_param != "" && retrieval_param_->Parse(retrieval_param.c_str())) {
    LOG(ERROR) << "parse retrieval param error";
    return -2;
  }

  std::map<string, int> vec_dups;
  for (int i = 0; i < vectors_num; i++) {
    string name(vectors_info[i]->name->value, vectors_info[i]->name->len);
    auto it = vec_dups.find(name);
    if (it == vec_dups.end()) {
      vec_dups[name] = 1;
    } else {
      ++vec_dups[name];
    }
  }

  RetrievalModel model = default_model_;
  if (!strcasecmp("IVFPQ", retrieval_type.c_str())) {
    model = RetrievalModel::IVFPQ;
  } else if (!strcasecmp("GPU", retrieval_type.c_str())) {
    model = RetrievalModel::GPU_IVFPQ;
  } else if (!strcasecmp("BINARYIVF", retrieval_type.c_str())) {
    model = RetrievalModel::BINARYIVF;
  } else if (!strcasecmp("HNSW", retrieval_type.c_str())) {
    model = RetrievalModel::HNSW;
  } else if (!strcasecmp("FLAT", retrieval_type.c_str())) {
    model = RetrievalModel::FLAT;
  } else {
    LOG(ERROR) << "NO support for retrieval type " << retrieval_type;
    return -1;
  }

  for (int i = 0; i < vectors_num; i++) {
    std::string vec_name(vectors_info[i]->name->value,
                         vectors_info[i]->name->len);
    int dimension = vectors_info[i]->dimension;

    std::string store_type_str(vectors_info[i]->store_type->value,
                               vectors_info[i]->store_type->len);

    VectorStorageType store_type = default_store_type_;
    if (store_type_str != "") {
      if (!strcasecmp("Mmap", store_type_str.c_str())) {
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

    std::string store_param;
    if (vectors_info[i]->store_param) {
      store_param.assign(vectors_info[i]->store_param->value,
                         vectors_info[i]->store_param->len);
    }

    if (model == RetrievalModel::BINARYIVF) {
      RawVector<uint8_t> *vec = RawVectorFactory::CreateBinary(
          store_type, vec_name, dimension / 8, max_doc_size_, root_path_,
          store_param);
      if (vec == nullptr) {
        LOG(ERROR) << "create raw vector error";
        return -1;
      }
      bool has_source = vectors_info[i]->has_source;
      bool multi_vids = vec_dups[vec_name] > 1 ? true : false;
      int ret = vec->Init(has_source, multi_vids);
      if (ret != 0) {
        LOG(ERROR) << "Raw vector " << vec_name << " init error, code [" << ret
                   << "]!";
        return -1;
      }

      StartFlushingIfNeed<uint8_t>(vec);
      raw_binary_vectors_[vec_name] = vec;

      GammaIndex *index =
          GammaIndexFactory::CreateBinary(model, dimension, docids_bitmap_, vec,
                                          retrieval_param, gamma_counters_);
      if (index == nullptr) {
        LOG(ERROR) << "create gamma index " << vec_name << " error!";
        return -1;
      }
      index->SetRawVectorBinary(vec);
      if (vectors_info[i]->is_index == FALSE) {
        LOG(INFO) << vec_name << " need not to indexed!";
        continue;
      }

      vector_indexes_[vec_name] = index;
    } else {
      RawVector<float> *vec =
          RawVectorFactory::Create(store_type, vec_name, dimension,
                                   max_doc_size_, root_path_, store_param);
      if (vec == nullptr) {
        LOG(ERROR) << "create raw vector error";
        return -1;
      }
      bool has_source = vectors_info[i]->has_source;
      bool multi_vids = vec_dups[vec_name] > 1 ? true : false;
      int ret = vec->Init(has_source, multi_vids);
      if (ret != 0) {
        LOG(ERROR) << "Raw vector " << vec_name << " init error, code [" << ret
                   << "]!";
        return -1;
      }

      StartFlushingIfNeed<float>(vec);
      raw_vectors_[vec_name] = vec;

      GammaIndex *index =
          GammaIndexFactory::Create(model, dimension, docids_bitmap_, vec,
                                    retrieval_param, gamma_counters_);
      if (index == nullptr) {
        LOG(ERROR) << "create gamma index " << vec_name << " error!";
        return -1;
      }
      index->SetRawVectorFloat(vec);
      if (vectors_info[i]->is_index == FALSE) {
        LOG(INFO) << vec_name << " need not to indexed!";
        continue;
      }

      vector_indexes_[vec_name] = index;
    }
  }
  table_created_ = true;
  return 0;
}

int VectorManager::AddToStore(int docid, std::vector<Field *> &fields) {
  for (unsigned int i = 0; i < fields.size(); i++) {
    std::string name =
        std::string(fields[i]->name->value, fields[i]->name->len);
    if (raw_vectors_.find(name) == raw_vectors_.end()) {
      // LOG(ERROR) << "Cannot find raw vector [" << name << "]";
      continue;
    }
    raw_vectors_[name]->Add(docid, fields[i]);
  }

  for (unsigned int i = 0; i < fields.size(); i++) {
    std::string name =
        std::string(fields[i]->name->value, fields[i]->name->len);
    if (raw_binary_vectors_.find(name) == raw_binary_vectors_.end()) {
      // LOG(ERROR) << "Cannot find raw vector [" << name << "]";
      continue;
    }
    raw_binary_vectors_[name]->Add(docid, fields[i]);
  }
  return 0;
}

int VectorManager::Update(int docid, std::vector<Field *> &fields) {
  for (unsigned int i = 0; i < fields.size(); i++) {
    string name = string(fields[i]->name->value, fields[i]->name->len);
    auto it = raw_vectors_.find(name);
    if (it == raw_vectors_.end()) {
      continue;
    }
    RawVector<float> *raw_vector = it->second;
    if (raw_vector->GetDimension() !=
        fields[i]->value->len / (int)sizeof(float)) {
      LOG(ERROR) << "invalid field value len=" << fields[i]->value->len
                 << ", dimension=" << raw_vector->GetDimension();
      return -1;
    }

    return raw_vector->Update(docid, fields[i]);
  }

  for (unsigned int i = 0; i < fields.size(); i++) {
    string name = string(fields[i]->name->value, fields[i]->name->len);
    auto it = raw_binary_vectors_.find(name);
    if (it == raw_binary_vectors_.end()) {
      continue;
    }
    RawVector<uint8_t> *raw_vector = it->second;
    if (raw_vector->GetDimension() !=
        fields[i]->value->len / (int)sizeof(float)) {
      LOG(ERROR) << "invalid field value len=" << fields[i]->value->len
                 << ", dimension=" << raw_vector->GetDimension();
      return -1;
    }

    return raw_vector->Update(docid, fields[i]);
  }
  return 0;
}

int VectorManager::Delete(int docid) {
  for (const auto &iter : vector_indexes_) {
    if (0 != iter.second->Delete(docid)) {
      LOG(ERROR) << "delete index from" << iter.first << " failed! docid=" << docid;
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
    if (0 != iter.second->AddRTVecsToIndex()) {
      ret = -1;
      LOG(ERROR) << "vector table " << iter.first
                 << " add real time vectors failed!";
    }
  }
  return ret;
}

int VectorManager::Search(const GammaQuery &query, GammaResult *results) {
  int ret = 0, n = 0;

  VectorResult all_vector_results[query.vec_num];

  query.condition->sort_by_docid = query.vec_num > 1 ? true : false;
  query.condition->metric_type =
      static_cast<DistanceMetricType>(retrieval_param_->metric_type);
  std::string vec_names[query.vec_num];
  for (int i = 0; i < query.vec_num; i++) {
    std::string name = std::string(query.vec_query[i]->name->value,
                                   query.vec_query[i]->name->len);
    vec_names[i] = name;
    std::map<std::string, GammaIndex *>::iterator iter =
        vector_indexes_.find(name);
    if (iter == vector_indexes_.end()) {
      LOG(ERROR) << "Query name " << name
                 << " not exist in created vector table";
      return -1;
    }

    GammaIndex *index = iter->second;
    int d = 0;
    if (index->raw_vec_binary_ != nullptr) {
      d = index->raw_vec_binary_->GetDimension();
      n = query.vec_query[i]->value->len / (sizeof(uint8_t) * d);
    } else {
      d = index->raw_vec_->GetDimension();
      n = query.vec_query[i]->value->len / (sizeof(float) * d);
    }

    if (n <= 0) {
      LOG(ERROR) << "Search n shouldn't less than 0!";
      return -1;
    }

    if (!all_vector_results[i].init(n, query.condition->topn)) {
      LOG(ERROR) << "Query name " << name << "init vector result error";
      return -1;
    }

    query.condition->min_dist = query.vec_query[i]->min_score;
    query.condition->max_dist = query.vec_query[i]->max_score;
    int ret_vec = index->Search(query.vec_query[i], query.condition,
                                all_vector_results[i]);
    if (ret_vec != 0) {
      ret = ret_vec;
    }
#ifdef PERFORMANCE_TESTING
    std::string msg;
    msg += "search " + std::to_string(i);
    query.condition->Perf(msg);
#endif
  }

  if (query.condition->sort_by_docid) {
    for (int i = 0; i < n; i++) {
      int start_docid = 0, common_docid_count = 0, common_idx = 0;
      double score = 0;
      bool has_common_docid = true;
      if (!results[i].init(query.condition->topn, vec_names, query.vec_num)) {
        LOG(ERROR) << "init gamma result(sort by docid) error, topn="
                   << query.condition->topn
                   << ", vector number=" << query.vec_num;
        return -1;
      }
      while (start_docid < INT_MAX) {
        for (int j = 0; j < query.vec_num; j++) {
          float vec_dist = 0;
          char *source = nullptr;
          int source_len = 0;
          int cur_docid = all_vector_results[j].seek(i, start_docid, vec_dist,
                                                     source, source_len);
          if (cur_docid == start_docid) {
            common_docid_count++;
            double field_score = query.vec_query[j]->has_boost == 1
                                     ? (vec_dist * query.vec_query[j]->boost)
                                     : vec_dist;
            score += field_score;
            results[i].docs[common_idx]->fields[j].score = field_score;
            results[i].docs[common_idx]->fields[j].source = source;
            results[i].docs[common_idx]->fields[j].source_len = source_len;
            if (common_docid_count == query.vec_num) {
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
          case InnerProduct:
            std::sort(results[i].docs, results[i].docs + common_idx,
                      InnerProductCmp);
            break;
          case L2:
            std::sort(results[i].docs, results[i].docs + common_idx, L2Cmp);
            break;
          default:
            LOG(ERROR) << "invalid metric_type="
                       << query.condition->metric_type;
        }
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      // double score = 0;
      if (!results[i].init(query.condition->topn, vec_names, query.vec_num)) {
        LOG(ERROR) << "init gamma result error, topn=" << query.condition->topn
                   << ", vector number=" << query.vec_num;
        return -1;
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

        score = query.vec_query[0]->has_boost == 1
                    ? (score * query.vec_query[0]->boost)
                    : score;

        results[i].docs[pos]->fields[0].score = score;
        results[i].docs[pos]->score = score;
        pos++;
      }
      results[i].results_count = pos;
    }
  }

#ifdef PERFORMANCE_TESTING
  query.condition->Perf("merge result");
#endif
  return ret;
}

int VectorManager::GetVector(
    const std::vector<std::pair<string, int>> &fields_ids,
    std::vector<string> &vec, bool is_bytearray) {
  for (const auto &pair : fields_ids) {
    const string &field = pair.first;
    const int id = pair.second;
    std::map<std::string, GammaIndex *>::iterator iter =
        vector_indexes_.find(field);
    if (iter == vector_indexes_.end()) {
      continue;
    }
    GammaIndex *gamma_index = iter->second;
    if (gamma_index->raw_vec_ != nullptr) {
      RawVector<float> *raw_vec = gamma_index->raw_vec_;
      if (raw_vec == nullptr) {
        LOG(ERROR) << "raw_vec is null!";
        return -1;
      }
      int vid = raw_vec->vid_mgr_->GetFirstVID(id);

      char *source = nullptr;
      int len = -1;
      int ret = raw_vec->GetSource(vid, source, len);

      if (ret != 0 || len < 0) {
        LOG(ERROR) << "Get source failed!";
        return -1;
      }

      ScopeVector<float> scope_vec;
      raw_vec->GetVector(vid, scope_vec);
      const float *feature = scope_vec.Get();
      string str_vec;
      if (is_bytearray) {
        int d = raw_vec->GetDimension();
        int d_byte = d * sizeof(float);

        char feat_source[sizeof(d) + d_byte + len];

        memcpy((void *)feat_source, &d_byte, sizeof(int));
        int cur = sizeof(d_byte);

        memcpy((void *)(feat_source + cur), feature, d_byte);
        cur += d_byte;

        memcpy((void *)(feat_source + cur), source, len);

        str_vec =
            string((char *)feat_source, sizeof(unsigned int) + d_byte + len);
      } else {
        for (int i = 0; i < raw_vec->GetDimension(); ++i) {
          str_vec += std::to_string(feature[i]) + ",";
        }
        str_vec.pop_back();
      }
      vec.emplace_back(std::move(str_vec));
    } else {
      RawVector<uint8_t> *raw_vec = gamma_index->raw_vec_binary_;
      if (raw_vec == nullptr) {
        LOG(ERROR) << "raw_vec is null!";
        return -1;
      }

      int vid = raw_vec->vid_mgr_->GetFirstVID(id);

      char *source = nullptr;
      int len = -1;
      int ret = raw_vec->GetSource(vid, source, len);

      if (ret != 0 || len < 0) {
        LOG(ERROR) << "Get source failed!";
        return -1;
      }

      ScopeVector<uint8_t> scope_vec;
      raw_vec->GetVector(vid, scope_vec);
      const uint8_t *feature = scope_vec.Get();
      string str_vec;
      if (is_bytearray) {
        int d = raw_vec->GetDimension();
        int d_byte = d * sizeof(uint8_t);

        char feat_source[sizeof(d) + d_byte + len];

        memcpy((void *)feat_source, &d_byte, sizeof(int));
        int cur = sizeof(d_byte);

        memcpy((void *)(feat_source + cur), feature, d_byte);
        cur += d_byte;

        memcpy((void *)(feat_source + cur), source, len);

        str_vec =
            string((char *)feat_source, sizeof(unsigned int) + d_byte + len);
      } else {
        for (int i = 0; i < raw_vec->GetDimension(); ++i) {
          str_vec += std::to_string(feature[i]) + ",";
        }
        str_vec.pop_back();
      }
      vec.emplace_back(std::move(str_vec));
    }
  }
  return 0;
}

int VectorManager::Dump(const string &path, int dump_docid, int max_docid) {
  for (const auto &iter : vector_indexes_) {
    const string &vec_name = iter.first;
    GammaIndex *index = iter.second;

    int max_vid = -1;
    auto it = raw_vectors_.find(vec_name);
    if (it == raw_vectors_.end()) {
      auto it2 = raw_binary_vectors_.find(vec_name);
      if (it2 == raw_binary_vectors_.end()) {
        LOG(ERROR) << "Cannot find vector [" << vec_name << "]";
        return -1;
      }
      max_vid = it2->second->vid_mgr_->GetLastVID(max_docid);
    } else {
      max_vid = it->second->vid_mgr_->GetLastVID(max_docid);
    }
    int dump_num = index->Dump(path, max_vid);
    if (dump_num < 0) {
      LOG(ERROR) << "vector " << vec_name << " dump gamma index failed!";
      return -1;
    }
    LOG(INFO) << "vector " << vec_name << " dump gamma index success!";
  }

  for (const auto &iter : raw_vectors_) {
    const string &vec_name = iter.first;
    RawVector<float> *raw_vector = iter.second;
    int ret = raw_vector->Dump(path, dump_docid, max_docid);
    if (ret != 0) {
      LOG(ERROR) << "vector " << vec_name << " dump failed!";
      return -1;
    }
    LOG(INFO) << "vector " << vec_name << " dump success!";
  }

  for (const auto &iter : raw_binary_vectors_) {
    const string &vec_name = iter.first;
    RawVector<uint8_t> *raw_vector = iter.second;
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

  for (const auto &iter : raw_binary_vectors_) {
    if (0 != iter.second->Load(index_dirs, doc_num)) {
      LOG(ERROR) << "vector [" << iter.first << "] load failed!";
      return -1;
    }
    LOG(INFO) << "vector [" << iter.first << "] load success!";
  }

  if (index_dirs.size() > 0) {
    for (const auto &iter : vector_indexes_) {
      if (iter.second->Load(index_dirs) < 0) {
        LOG(ERROR) << "vector [" << iter.first << "] load gamma index failed!";
        return -1;
      } else {
        LOG(INFO) << "vector [" << iter.first << "] load gamma index success!";
      }
    }
  }

  return 0;
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

  for (const auto &iter : raw_binary_vectors_) {
    if (iter.second != nullptr) {
      StopFlushingIfNeed(iter.second);
      delete iter.second;
    }
  }
  raw_binary_vectors_.clear();

  for (const auto &iter : vector_indexes_) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
  vector_indexes_.clear();
  LOG(INFO) << "Vector indexes cleared.";

  if (retrieval_param_ != nullptr) {
    delete retrieval_param_;
    retrieval_param_ = nullptr;
  }
  LOG(INFO) << "VectorManager closed.";
}
}  // namespace tig_gamma
