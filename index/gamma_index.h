#pragma once

#include <faiss/Index.h>

#include "index/impl/gamma_index_flat.h"
#include "index/impl/gamma_index_ivfflat.h"
#include "index/impl/relayout/gamma_index_ivfpq_relayout.h"
#include "index/impl/relayout/x86/x86_gamma_index_ivfflat.h"
#include "index/retrieval_model.h"
#include "util/bitmap_manager.h"
#include "vector/raw_vector.h"

namespace tig_gamma {

using idx_t = faiss::Index::idx_t;

/**
 * faiss_like index
 *
 */
class Index {
 public:
  /**
   * @brief Construct a new Gamma Faisslike Index object
   *
   * @param retrieval_type : IVFFLAT, IVFPQ
   * @param d : dimension
   * @param metric : support METRIC_INNER_PRODUCT, METRIC_L2
   */
  Index();

  virtual ~Index();

  /**
   * @brief init index by json string
   *
   * @param index_param : example "{\"nprobe\" : 10, \"ncentroids\" : 256
   * ,\"nsubvector\" : 64}"
   * @return int
   */
  virtual int init(const std::string &index_param) { return 0; };

  virtual int init() { return 0; };

  virtual void train(idx_t n, const float *x){};

  virtual void add(idx_t n, const float *x){};

  virtual void search(idx_t n, const float *x, idx_t k, float *distances,
                      idx_t *labels){};

  virtual int dump(const std::string &dir) { return 0; };

  virtual int load(const std::string &dir) { return 0; };

 protected:
  bitmap::BitmapManager *docids_bitmap_;
  RawVector *raw_vector_;
  std::string index_param;
  std::string vec_name;
};

class IndexIVFFlat : public x86GammaIndexIVFFlat, public Index {
 public:
  IndexIVFFlat(faiss::Index *quantizer, size_t d, size_t nlist,
               faiss::MetricType metric = faiss::METRIC_L2);

  virtual ~IndexIVFFlat();

  int init(const std::string &index_param);

  int init();

  void train(idx_t n, const float *x) override;

  void add(idx_t n, const float *x) override;

  void search(idx_t n, const float *x, idx_t k, float *distances,
              idx_t *labels);

  int dump(const std::string &dir);

  int load(const std::string &dir);
};

class IndexIVFPQ : public GammaIndexIVFPQRelayout, public Index {
 public:
  IndexIVFPQ(faiss::Index *quantizer, size_t d, size_t nlist, size_t M,
             size_t nbits_per_idx, faiss::MetricType metric = faiss::METRIC_L2);

  virtual ~IndexIVFPQ();

  int init(const std::string &index_param);

  int init();

  void train(idx_t n, const float *x) override;

  void add(idx_t n, const float *x) override;

  void search(idx_t n, const float *x, idx_t k, float *distances,
              idx_t *labels);

  int dump(const std::string &dir);

  int load(const std::string &dir);
};

Index *index_factory(int d, const char *description_in,
                     faiss::MetricType metric);

}  // namespace tig_gamma
