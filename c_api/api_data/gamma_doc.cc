/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "gamma_doc.h"

namespace tig_gamma {

int Doc::Serialize(char **out, int *out_len) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<gamma_api::Field>> field_vector(fields_.size());

  int i = 0;
  for (const struct Field &f : fields_) {
    std::vector<uint8_t> value(f.value.size());
    memcpy(value.data(), f.value.data(), f.value.size());

    auto field = gamma_api::CreateField(
        builder, builder.CreateString(f.name), builder.CreateVector(value),
        builder.CreateString(f.source),
        static_cast<::DataType>(f.datatype));
    field_vector[i++] = field;
  }
  auto field_vec = builder.CreateVector(field_vector);
  auto doc = gamma_api::CreateDoc(builder, field_vec);
  builder.Finish(doc);
  *out_len = builder.GetSize();
  *out = (char *)malloc(*out_len * sizeof(char));
  memcpy(*out, (char *)builder.GetBufferPointer(), *out_len);
  return 0;
}

void Doc::Deserialize(const char *data, int len) {
  doc_ = const_cast<gamma_api::Doc *>(gamma_api::GetDoc(data));

  size_t fields_num = doc_->fields()->size();
  fields_.resize(fields_num);
  for (size_t i = 0; i < fields_num; ++i) {
    auto f = doc_->fields()->Get(i);
    struct Field field;
    field.name = f->name()->str();
    field.value = std::string(
        reinterpret_cast<const char *>(f->value()->Data()), f->value()->size());
    field.source = f->source()->str();
    field.datatype = static_cast<DataType>(f->data_type());

    fields_[i] = std::move(field);
  }
}

void Doc::AddField(const struct Field &field) { fields_.push_back(field); }

void Doc::AddField(struct Field &&field) { fields_.emplace_back(std::forward<struct Field>(field)); }

std::vector<struct Field> &Doc::Fields() { return fields_; }

}  // namespace tig_gamma