/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "vector_block.h"

#include <unistd.h>

namespace tig_gamma {

VectorBlock::VectorBlock(int fd, int per_block_size, int length,
                         uint32_t header_size, uint32_t seg_id,
                         uint32_t seg_block_capacity)
    : Block(fd, per_block_size, length, header_size, seg_id,
            seg_block_capacity) {
  vec_item_len_ = item_length_;
  LOG(INFO) << "VectorBlock construction!";
}

void VectorBlock::InitSubclass() {
  if(compressor_) {
    vec_item_len_ = compressor_->GetCompressLen();
    LOG(INFO) << "Vector block use compress. vec_item_len_[" 
              << vec_item_len_ << "]";
    if (compressor_->GetCompressType() != CompressType::Zfp) {
      LOG(ERROR) << "The compression method used by vec_block is not ZFP.";
    }
  }
}

int VectorBlock::GetReadFunParameter(ReadFunParameter &parameter, uint32_t len,
                                     uint32_t off) {
  parameter.fd = fd_;
  parameter.len = len;
  parameter.offset = off;
  parameter.cmprsr = (void*)compressor_;
#ifdef WITH_ZFP
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    parameter.offset = (parameter.offset / raw_len) * vec_item_len_;
    parameter.len = (parameter.len / raw_len) * vec_item_len_;
  }
#endif
  parameter.offset += header_size_;
  return 0;
}

bool VectorBlock::ReadBlock(uint32_t key,
                            std::shared_ptr<std::vector<uint8_t>> &block,
                            ReadFunParameter *param) {
#ifdef WITH_ZFP
  Compressor *compressor = (Compressor *)param->cmprsr;
  if (compressor) {
    char *cmprs_data = new char[param->len];
    int cmprs_len = compressor->GetCompressLen();
    int batch_num = param->len / cmprs_len;
    block = std::make_shared<std::vector<uint8_t>>(batch_num * compressor->GetRawLen());
    pread(param->fd, cmprs_data, param->len, param->offset);
    if (batch_num == 1) {
      compressor->Decompress((char *)cmprs_data, (char *)(block->data()), param->len);
    } else {
      compressor->DecompressBatch((char *)cmprs_data, (char *)(block->data()),
                                  batch_num, param->len);
    }
    delete[] cmprs_data;
  } else 
#endif
  {
    block = std::make_shared<std::vector<uint8_t>>(param->len);
    pread(param->fd, block->data(), param->len, param->offset);
  }
  return true;
}

int VectorBlock::WriteContent(const uint8_t *data, int len, uint32_t offset,
                              disk_io::AsyncWriter *disk_io) {
#ifdef WITH_ZFP
  const uint8_t *raw_val = data;
  std::vector<uint8_t> cmprs_val;
  if (compressor_) {
    int raw_len = len;
    len = vec_item_len_;
    offset = (offset / raw_len) * len;
    cmprs_val.resize(len);
    compressor_->Compress((char *)raw_val, (char *)cmprs_val.data(),
                              raw_len);
    data = (const uint8_t *)cmprs_val.data();
  }
#endif

  disk_io->Set(header_size_, vec_item_len_);
  struct disk_io::WriterStruct *write_struct = new struct disk_io::WriterStruct;
  write_struct->fd = fd_;
  write_struct->data = new uint8_t[len];
  memcpy(write_struct->data, data, len);
  write_struct->start = header_size_ + offset;
  write_struct->len = len;
  disk_io->AsyncWrite(write_struct);
  // disk_io->SyncWrite(write_struct);
  return 0;
}

int VectorBlock::ReadContent(uint8_t *value, uint32_t len, uint32_t offset) {
#ifdef WITH_ZFP
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    int batch_num = len / raw_len;
    int cmprs_data_len = batch_num * vec_item_len_;
    char *cmprs_data = new char[cmprs_data_len];
    offset = (offset / raw_len) * vec_item_len_;
    pread(fd_, cmprs_data, cmprs_data_len, header_size_ + offset);

    if (batch_num == 1) {
      compressor_->Decompress((char *)cmprs_data, (char *)value, len);
    } else {
      compressor_->DecompressBatch((char *)cmprs_data, (char *)value, batch_num,
                                       len);
    }
    delete[] cmprs_data;
  } else
#endif
  {
    pread(fd_, value, len, header_size_ + offset);
  }
  return 0;
}

int VectorBlock::SubclassUpdate(const uint8_t *data, int len, uint32_t offset) {
#ifdef WITH_ZFP
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    int batch_num = len / raw_len;
    int cmprs_data_len = batch_num * vec_item_len_;
    char *cmprs_data = new char[cmprs_data_len];
    offset = (offset / raw_len) * vec_item_len_;

    if (batch_num == 1) {
      compressor_->Compress((char *)data, (char *)cmprs_data, len);
    } else {
      compressor_->CompressBatch((char *)data, (char *)cmprs_data, batch_num,
                                 len);
    }
    pwrite(fd_, cmprs_data, cmprs_data_len, header_size_ + offset);
    delete[] cmprs_data;
    cmprs_data = nullptr;
  } else
#endif
  {
    pwrite(fd_, data, len, header_size_ + offset);
  }
  return 0;
}

}  // namespace tig_gamma