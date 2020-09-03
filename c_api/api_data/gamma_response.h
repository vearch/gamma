/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "gamma_common_data.h"
#include "gamma_raw_data.h"
#include "response_generated.h"

namespace tig_gamma {

enum class SearchResultCode : std::uint16_t {
  SUCCESS = 0,
  INDEX_NOT_TRAINED,
  SEARCH_ERROR
};

struct ResultItem {
  ResultItem() { score = -1; }

  ResultItem(const ResultItem &other) {
    score = other.score;
    names = other.names;
    values = other.values;
    extra = other.extra;
  }

  ResultItem &operator=(const ResultItem &other) {
    score = other.score;
    names = other.names;
    values = other.values;
    extra = other.extra;
    return *this;
  }

  ResultItem(ResultItem &&other) {
    score = other.score;
    names = std::move(other.names);
    values = std::move(other.values);
    extra = std::move(other.extra);
  }

  ResultItem &operator=(ResultItem &&other) {
    score = other.score;
    names = std::move(other.names);
    values = std::move(other.values);
    extra = std::move(other.extra);
    return *this;
  }

  double score;
  std::vector<std::string> names;
  std::vector<std::string> values;
  std::string extra;
};

struct SearchResult {
  SearchResult() { total = 0; }

  SearchResult(const SearchResult &other) {
    total = other.total;
    result_code = other.result_code;
    msg = other.msg;
    result_items = other.result_items;
  }

  SearchResult &operator=(const SearchResult &other) {
    total = other.total;
    result_code = other.result_code;
    msg = other.msg;
    result_items = other.result_items;
    return *this;
  }

  SearchResult(SearchResult &&other) {
    total = other.total;
    result_code = other.result_code;
    msg = std::move(other.msg);
    result_items = std::move(other.result_items);
  }

  SearchResult &operator=(SearchResult &&other) {
    total = other.total;
    result_code = other.result_code;
    msg = std::move(other.msg);
    result_items = std::move(other.result_items);
    return *this;
  }

  int total;
  SearchResultCode result_code;
  std::string msg;
  std::vector<struct ResultItem> result_items;
};

class Response : public RawData {
 public:
  Response();

  virtual int Serialize(char **out, int *out_len);

  virtual void Deserialize(const char *data, int len);

  void AddResults(const struct SearchResult &result);

  void AddResults(struct SearchResult &&result);

  std::vector<struct SearchResult> &Results();

  void SetOnlineLogMessage(const std::string &msg);

 private:
  gamma_api::Response *response_;
  std::vector<struct SearchResult> results_;
  std::string online_log_message_;
};

}  // namespace tig_gamma