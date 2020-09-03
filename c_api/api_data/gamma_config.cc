/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_config.h"

namespace tig_gamma {

int Config::Serialize(char **out, int *out_len) {
  flatbuffers::FlatBufferBuilder builder;
  auto config =
      gamma_api::CreateConfig(builder, builder.CreateString(path_),
                              builder.CreateString(log_dir_));
  builder.Finish(config);
  *out_len = builder.GetSize();
  *out = (char *)malloc(*out_len * sizeof(char));
  memcpy(*out, (char *)builder.GetBufferPointer(), *out_len);
  return 0;
}

void Config::Deserialize(const char *data, int len) {
  config_ = const_cast<gamma_api::Config *>(gamma_api::GetConfig(data));

  path_ = config_->path()->str();
  log_dir_ = config_->log_dir()->str();
}

const std::string &Config::Path() {
  assert(config_);
  return path_;
}

void Config::SetPath(std::string &path) { path_ = path; }

const std::string &Config::LogDir() {
  assert(config_);
  return log_dir_;
}

void Config::SetLogDir(std::string &log_dir) { log_dir_ = log_dir; }

}  // namespace tig_gamma