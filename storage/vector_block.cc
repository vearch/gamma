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
                         uint32_t seg_block_capacity,
                         const std::atomic<uint32_t> *cur_size, int max_size)
    : Block(fd, per_block_size, length, header_size, seg_id,
            seg_block_capacity, cur_size, max_size) {
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
  parameter.offset += header_size_;
  return 0;
}

bool VectorBlock::ReadBlock(uint32_t key, char *block,
                            ReadFunParameter *param) {
  if (param->len > MAX_BLOCK_SIZE) {
    LOG(ERROR) << "vector ReadConten len is:" << param->len
               << " key:" << key;
    return false;
  }
  if (block == nullptr) {
    LOG(ERROR) << "ReadString block is nullptr.";
    return false;
  }
  pread(param->fd, block, param->len, param->offset);
  return true;
}

int VectorBlock::WriteContent(const uint8_t *data, int len, uint32_t offset,
                              disk_io::AsyncWriter *disk_io, 
                              std::atomic<uint32_t> *cur_size) {
#ifdef WITH_ZFP
  std::vector<char> output;
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    Compress(data, len, output);

    offset = (offset / raw_len) * vec_item_len_;
    data = (const uint8_t *)output.data();
  }
#endif

  disk_io->Set(header_size_, vec_item_len_);
  struct disk_io::WriterStruct *write_struct = new struct disk_io::WriterStruct;
  write_struct->fd = fd_;
  write_struct->data = new uint8_t[vec_item_len_];
  memcpy(write_struct->data, data, vec_item_len_);
  write_struct->start = header_size_ + offset;
  write_struct->len = vec_item_len_;
  write_struct->cur_size = cur_size;
  disk_io->AsyncWrite(write_struct);
  // disk_io->SyncWrite(write_struct);
  return 0;
}

int VectorBlock::ReadContent(uint8_t *value, uint32_t len, uint32_t offset) {
#ifdef WITH_ZFP
  if (compressor_) {
    uint32_t raw_len = (uint32_t)(compressor_->GetRawLen());
    uint32_t batch_num = len / raw_len;
    uint32_t cmprs_data_len = batch_num * vec_item_len_;
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

int VectorBlock::Read(uint8_t *value, uint32_t n_bytes, uint32_t start) {
  if (lru_cache_ == nullptr) {
    return ReadContent(value, n_bytes, start);
  }

#ifdef WITH_ZFP
  uint32_t raw_len = 0;
  if (compressor_) {
    raw_len = (uint32_t)(compressor_->GetRawLen());
    n_bytes = (n_bytes / raw_len) * vec_item_len_;
    start = (start / raw_len) * vec_item_len_;
  }
#endif

  int read_num = 0;
  while (n_bytes) {
    int len = n_bytes;
    if (len > per_block_size_) len = per_block_size_;

    uint32_t block_id = start / per_block_size_;
    // uint32_t block_pos = block_pos_[block_id];
    uint32_t block_pos = block_id * per_block_size_;
    uint32_t block_offset = start % per_block_size_;

    if (len > per_block_size_ - block_offset)
      len = per_block_size_ - block_offset;

    if (last_bid_in_disk_ <= block_id) {
      last_bid_in_disk_ = (*cur_size_) * item_length_ / per_block_size_;
      if ((int)(*cur_size_) == max_size_) SegmentIsFull();
    }
    if (last_bid_in_disk_ <= block_id) {
#ifdef WITH_ZFP
      if (compressor_) {
        uint8_t *output = value + (read_num / vec_item_len_) * raw_len;
        uint32_t read_len = (len / vec_item_len_) * raw_len;
        uint32_t offset = ((block_pos + block_offset) / vec_item_len_) * raw_len;
        ReadContent(output, read_len, offset);
      } else
#endif
      {
        ReadContent(value + read_num, len, block_pos + block_offset);
      }
    } else {
      // std::shared_ptr<std::vector<uint8_t>> block;
      char *block = nullptr;
      uint32_t cache_bid = GetCacheBlockId(block_id);
      bool res = lru_cache_->Get(cache_bid, block);
      if (not res) {
        ReadFunParameter parameter;
        GetReadFunParameter(parameter, per_block_size_, block_pos);

        res = lru_cache_->SetOrGet(cache_bid, block, &parameter);
      }
      if (not res || block == nullptr) {
        LOG(ERROR) << "Read block fails from disk_file, block_id[" << block_id
                   << "]";
        return -1;
      }

      // char *block = nullptr;
      // uint32_t cache_bid = GetCacheBlockId(block_id);
      // bool res = lru_cache_->Get2(cache_bid, block);
      // if (block == nullptr) {
      //   LOG(ERROR) << "VectorBlock Get block=nullptr:";
      // }
      // if (not res) {
      //   ReadFunParameter parameter;
      //   GetReadFunParameter(parameter, per_block_size_, block_pos);
      //   pread(parameter.fd, block, parameter.len, parameter.offset);

      //   lru_cache_->Set2(cache_bid, block);
      //   if (block == nullptr) {
      //     LOG(ERROR) << "vector block cell = nullptr";
      //   }
      // }


#ifdef WITH_ZFP
      if (compressor_) {
        int batch_num = len / vec_item_len_;
        char *output = (char*)value + (read_num / vec_item_len_) * raw_len;
        if (batch_num == 1) {
          compressor_->Decompress(block + block_offset, output , 0);
        } else {
          compressor_->DecompressBatch(block + block_offset, output, batch_num, 0);
        }
      } else
#endif
      {
        memcpy(value + read_num, block + block_offset, len);
      }
    }

    start += len;
    read_num += len;
    n_bytes -= len;
  }
  return 0;
}

int VectorBlock::Compress(const uint8_t *data, int len, std::vector<char> &output) {
#ifdef WITH_ZFP
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    int batch_num = len / raw_len;
    int cmprs_data_len = batch_num * vec_item_len_;
    output.resize(cmprs_data_len);
    char *cmprs_data = output.data();

    if (batch_num == 1) {
      compressor_->Compress((char *)data, (char *)cmprs_data, 0);
    } else {
      compressor_->CompressBatch((char *)data, (char *)cmprs_data, batch_num,
                                 0);
    }
    return 0;
  }
#endif
  return -1;
}

int VectorBlock::Update(const uint8_t *data, int n_bytes, uint32_t offset) {
#ifdef WITH_ZFP
  std::vector<char> output;
  if (compressor_) {
    int raw_len = compressor_->GetRawLen();
    offset = (offset / raw_len) * vec_item_len_;
    Compress(data, n_bytes, output);
    data = (uint8_t *)output.data();
    n_bytes = output.size();
  }
#endif

  pwrite(fd_, data, n_bytes, header_size_ + offset);
  
  if (lru_cache_ == nullptr) {
    return 0;
  }
  while (n_bytes) {
    int len = n_bytes;
    if (len > per_block_size_) len = per_block_size_;

    uint32_t block_id = offset / per_block_size_;
    uint32_t block_offset = offset % per_block_size_;

    if (len > per_block_size_ - block_offset)
      len = per_block_size_ - block_offset;

    uint32_t cache_block_id = seg_id_ * seg_block_capacity_ + block_id;
    lru_cache_->Evict(cache_block_id);

    offset += len;
    n_bytes -= len;
  }
  return 0;
}

}  // namespace tig_gamma
