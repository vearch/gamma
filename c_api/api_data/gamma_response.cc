/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_response.h"

namespace tig_gamma {

Response::Response() { response_ = nullptr; }

int Response::Serialize(char **out, int *out_len) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<gamma_api::SearchResult>> search_results;
  for (size_t i = 0; i < results_.size(); ++i) {
    struct SearchResult &result = results_[i];

    std::vector<flatbuffers::Offset<gamma_api::ResultItem>> result_items;
    for (size_t j = 0; j < result.result_items.size(); ++j) {
      struct ResultItem &result_item = result.result_items[j];
      double score = result_item.score;

      std::vector<flatbuffers::Offset<gamma_api::Attribute>> attributes;
      for (size_t k = 0; k < result_item.names.size(); ++k) {
        std::vector<uint8_t> it(result_item.values[k].size());
        memcpy(it.data(), result_item.values[k].data(),
               result_item.values[k].size());
        attributes.emplace_back(gamma_api::CreateAttribute(
            builder, builder.CreateString(result_item.names[k]),
            builder.CreateVector(it)));
      }

      std::string &extra = result_item.extra;
      result_items.emplace_back(gamma_api::CreateResultItem(
          builder, score, builder.CreateVector(attributes),
          builder.CreateString(extra)));
    }

    auto item_vec = builder.CreateVector(result_items);

    auto msg = builder.CreateString(result.msg);
    gamma_api::SearchResultCode result_code =
        static_cast<gamma_api::SearchResultCode>(result.result_code);
    auto results = gamma_api::CreateSearchResult(builder, result.total,
                                                 result_code, msg, item_vec);
    search_results.push_back(results);
  }

  auto result_vec = builder.CreateVector(search_results);

  flatbuffers::Offset<flatbuffers::String> message =
      builder.CreateString(online_log_message_);
  auto res = gamma_api::CreateResponse(builder, result_vec, message);
  builder.Finish(res);

  *out_len = builder.GetSize();
  *out = (char *)malloc(*out_len * sizeof(char));
  memcpy(*out, (char *)builder.GetBufferPointer(), *out_len);
  return 0;
}

void Response::Deserialize(const char *data, int len) {
  assert(response_ == nullptr);
  response_ = const_cast<gamma_api::Response *>(gamma_api::GetResponse(data));
  size_t result_num = response_->results()->size();
  results_.resize(result_num);

  for (size_t i = 0; i < result_num; ++i) {
    SearchResult result;
    auto fbs_result = response_->results()->Get(i);
    result.total = fbs_result->total();
    result.result_code =
        static_cast<SearchResultCode>(fbs_result->result_code());
    result.msg = fbs_result->msg()->str();

    size_t items_num = fbs_result->result_items()->size();
    result.result_items.resize(items_num);

    for (size_t j = 0; j < items_num; ++j) {
      auto fbs_result_item = fbs_result->result_items()->Get(j);
      struct ResultItem item;
      item.score = fbs_result_item->score();
      size_t attr_num = fbs_result_item->attributes()->size();
      item.names.resize(attr_num);
      item.values.resize(attr_num);

      for (size_t k = 0; k < attr_num; ++k) {
        auto attr = fbs_result_item->attributes()->Get(k);
        item.names[k] = attr->name()->str();

        std::string item_value =
            std::string(reinterpret_cast<const char *>(attr->value()->Data()),
                        attr->value()->size());
        item.values[k] = std::move(item_value);
      }

      result.result_items[j] = std::move(item);
    }
    results_[i] = std::move(result);
  }

  online_log_message_ = response_->online_log_message()->str();
}

void Response::AddResults(const struct SearchResult &result) {
  results_.emplace_back(result);
}

void Response::AddResults(struct SearchResult &&result) {
  results_.emplace_back(std::forward<struct SearchResult>(result));
}

std::vector<struct SearchResult> &Response::Results() { return results_; }

void Response::SetOnlineLogMessage(const std::string &msg) {
  online_log_message_ = msg;
}
}  // namespace tig_gamma
