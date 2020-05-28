/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifndef GAMMA_INDEX_FACTORY_H_
#define GAMMA_INDEX_FACTORY_H_

#include "faiss/IndexBinaryFlat.h"
#include "gamma_common_data.h"
#include "gamma_index_binary_ivf.h"
#include "gamma_index_ivfpq.h"
#ifdef BUILD_GPU
#include "gamma_index_ivfpq_gpu.h"
#endif
#include "faiss/IndexFlat.h"
#include "gamma_index_flat.h"
#include "gamma_index_hnsw.h"
#include "mmap_raw_vector.h"
#include "raw_vector.h"

namespace tig_gamma {
class GammaIndexFactory {
 public:
  static GammaIndex *Create(RetrievalModel model, size_t dimension,
                            const char *docids_bitmap,
                            RawVector<float> *raw_vec,
                            std::string &retrieval_parameters,
                            GammaCounters *counters) {
    if (docids_bitmap == nullptr) {
      LOG(ERROR) << "docids_bitmap is NULL!";
      return nullptr;
    }
    LOG(INFO) << "Create index model [" << model << "]";

    switch (model) {
      case IVFPQ: {
        IVFPQRetrievalParams *ivfpq_param = new IVFPQRetrievalParams();
        if (retrieval_parameters != "" &&
            ivfpq_param->Parse(retrieval_parameters.c_str())) {
          delete ivfpq_param;
          return nullptr;
        }
        LOG(INFO) << ivfpq_param->ToString();
        if (dimension % ivfpq_param->nsubvector != 0) {
          dimension = (dimension / ivfpq_param->nsubvector + 1) *
                      ivfpq_param->nsubvector;
          LOG(INFO) << "Dimension [" << raw_vec->GetDimension()
                    << "] cannot divide by nsubvector ["
                    << ivfpq_param->nsubvector << "], adjusted to ["
                    << dimension << "]";
        }

        faiss::IndexFlatL2 *coarse_quantizer =
            new faiss::IndexFlatL2(dimension);
        GammaIndex *gamma_index = new GammaIVFPQIndex(
            coarse_quantizer, dimension, ivfpq_param->ncentroids,
            ivfpq_param->nsubvector, ivfpq_param->nbits_per_idx, docids_bitmap,
            raw_vec, counters);
        delete ivfpq_param;
        return gamma_index;
        break;
      }
#ifdef BUILD_GPU
      case GPU_IVFPQ: {
        IVFPQRetrievalParams *ivfpq_param = new IVFPQRetrievalParams();
        if (retrieval_parameters != "" &&
            ivfpq_param->Parse(retrieval_parameters.c_str())) {
          delete ivfpq_param;
          return nullptr;
        }
        LOG(INFO) << ivfpq_param->ToString();
        if (dimension % ivfpq_param->nsubvector != 0) {
          dimension = (dimension / ivfpq_param->nsubvector + 1) *
                      ivfpq_param->nsubvector;
          LOG(INFO) << "Dimension [" << raw_vec->GetDimension()
                    << "] cannot divide by nsubvector ["
                    << ivfpq_param->nsubvector << "], adjusted to ["
                    << dimension << "]";
        }

        gamma_gpu::GammaIVFPQGPUIndex *gpu_index =
            new gamma_gpu::GammaIVFPQGPUIndex(
                dimension, ivfpq_param->ncentroids, ivfpq_param->nsubvector,
                ivfpq_param->nbits_per_idx, docids_bitmap, raw_vec, counters);
        delete ivfpq_param;
        return (GammaIndex *)gpu_index;
        break;
      }
#endif

      case FLAT: {
        auto raw_vec_type = dynamic_cast<MmapRawVector<float> *>(raw_vec);
        if (raw_vec_type == nullptr || raw_vec_type->GetMemoryMode() == 0) {
          LOG(ERROR) << "FLAT cann't work in RocksDB or disk mode";
          return nullptr;
        }

        RetrievalParams *retrieval_param = new RetrievalParams();
        if (retrieval_parameters != "" &&
            retrieval_param->Parse(retrieval_parameters.c_str())) {
          delete retrieval_param;
          return nullptr;
        }
        LOG(INFO) << retrieval_param->ToString();
        GammaIndex *gamma_index =
            new GammaFLATIndex(dimension, docids_bitmap, raw_vec);
        delete retrieval_param;
        return gamma_index;
        break;
      }

      case HNSW: {
        auto raw_vec_type = dynamic_cast<MmapRawVector<float> *>(raw_vec);
        if (raw_vec_type == nullptr || raw_vec_type->GetMemoryMode() == 0) {
          LOG(ERROR) << "HNSW cann't work in RocksDB or disk mode";
          return nullptr;
        }

        HNSWRetrievalParams *hnsw_param = new HNSWRetrievalParams();
        if (retrieval_parameters != "" &&
            hnsw_param->Parse(retrieval_parameters.c_str())) {
          delete hnsw_param;
          return nullptr;
        }
        LOG(INFO) << hnsw_param->ToString();

        GammaIndex *gamma_index = new GammaHNSWFlatIndex(
            dimension, hnsw_param->metric_type, hnsw_param->nlinks,
            hnsw_param->efSearch, hnsw_param->efConstruction, docids_bitmap,
            raw_vec);
        delete hnsw_param;
        return gamma_index;
        break;
      }

      default: {
        LOG(ERROR) << "invalid retrieval model type";
        break;
      }
    }

    return nullptr;
  };

  static GammaIndex *CreateBinary(RetrievalModel model, size_t dimension,
                                  const char *docids_bitmap,
                                  RawVector<uint8_t> *raw_vec,
                                  std::string &retrieval_parameters,
                                  GammaCounters *counters) {
    if (docids_bitmap == nullptr) {
      LOG(ERROR) << "docids_bitmap is NULL!";
      return nullptr;
    }
    LOG(INFO) << "Create index model [" << model << "]";

    BinaryRetrievalParams binary_param;
    if (retrieval_parameters != "" &&
        binary_param.Parse(retrieval_parameters.c_str())) {
      return nullptr;
    }

    LOG(INFO) << binary_param.ToString();

    switch (model) {
      case BINARYIVF: {
        faiss::IndexBinaryFlat *coarse_quantizer =
            new faiss::IndexBinaryFlat(dimension);
        GammaIndex *gamma_index = (GammaIndex *)new GammaIndexBinaryIVF(
            coarse_quantizer, dimension, binary_param.ncentroids,
            binary_param.nprobe, docids_bitmap, raw_vec);
        return gamma_index;
      }
      default: {
        throw std::invalid_argument("invalid raw feature type");
        break;
      }
    }

    return nullptr;
  }
};

}  // namespace tig_gamma

#endif  // GAMMA_INDEX_FACTORY_H_
