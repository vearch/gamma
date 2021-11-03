#pragma once

#include <string>

namespace tig_gamma {

struct TermFilter {
  std::string field;
  std::string value;
  int is_union;
};

struct RangeFilter {
  std::string field;
  std::string lower_value;
  std::string upper_value;
  bool include_lower;
  bool include_upper;
};

struct VectorQuery {
  std::string name;
  std::string value;
  double min_score;
  double max_score;
  double boost;
  int has_boost;
  std::string retrieval_type;
};

}
