/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>
#include <unistd.h>

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "log.h"

struct ReadFunParameter {
  int fd;
  uint32_t len;
  uint32_t offset;
  void *cmprs;
};

struct ReadStrFunParameter {
  int fd;
  uint32_t block_id;
  uint32_t in_block_pos;
  uint32_t len;
  void *str_block;
};

#define THRESHOLD_OF_SWAP 250
#define THRESHOLD_TYPE uint8_t
#define MAP_GROUP_NUM 100

template <typename Value>
class CacheQueue {
 public:
  struct Node {
    Value val;
    Node *prev;
    Node *next;
  };
  CacheQueue() {
    head_ = nullptr;
    tail_ = nullptr;
  }
  ~CacheQueue() {
    while (head_) {
      Node *del = head_;
      head_ = del->next;
      delete del;
    }
    tail_ = nullptr;
  }

  void Init() {
    head_ = new Node;
    tail_ = head_;
    head_->next = nullptr;
    head_->prev = nullptr;
  }

  void Erase(void *n) {
    if (!n) {
      return;
    }
    Node *del = (Node *)n;
    del->prev->next = del->next;
    if (del->next) {
      del->next->prev = del->prev;
    } else {
      tail_ = del->prev;
    }
    delete del;
  }

  void MoveToTail(void *n) {
    Node *node = (Node *)n;
    if (!node || node->prev == nullptr || node->next == NULL || node == tail_) {
      return;
    }
    node->prev->next = node->next;
    node->next->prev = node->prev;
    tail_->next = node;
    node->prev = tail_;
    node->next = nullptr;
    tail_ = node;
  }

  void *Insert(Value value) {
    tail_->next = new Node;
    tail_->next->prev = tail_;
    tail_ = tail_->next;
    tail_->val = value;
    tail_->next = nullptr;

    return (void *)tail_;
  }

  bool Pop(Value &value) {
    if (head_ != tail_) {
      Node *del = head_->next;
      value = del->val;
      head_->next = del->next;
      if (del->next) {
        del->next->prev = head_;
      } else {
        tail_ = head_;
      }
      delete del;
      return true;
    }
    return false;
  }

 private:
  Node *head_;
  Node *tail_;
};

template <typename Key, typename Mapped, typename FuncToken,
          typename HashFunction = std::hash<Key>>
class LRUCache {
 public:
  using LoadFunc = bool (*)(Key, std::shared_ptr<Mapped> &, FuncToken);

  struct Cell {
    std::shared_ptr<Mapped> value;
    void *queue_ite;
    THRESHOLD_TYPE hits;
  };

  struct InsertInfo {
    explicit InsertInfo(LRUCache &cache) : lru_cache_(cache) {}
    std::mutex mtx_;
    bool is_clean_ = false;
    bool is_product_ = false;
    std::shared_ptr<Mapped> value_;
    LRUCache &lru_cache_;
  };

  class CellsGroup {
   public:
    std::unordered_map<Key, Cell, HashFunction> cells_;
    std::unordered_map<Key, std::shared_ptr<InsertInfo>, HashFunction>
        insert_infos_;
  };

 private:
  size_t max_size_;
  std::unordered_map<Key, std::shared_ptr<InsertInfo>, HashFunction>
      insert_infos_;
  size_t max_overflow_ = 0;
  size_t last_show_log_ = 1;
  std::atomic<size_t> cur_size_{0};
  std::atomic<size_t> hits_{0};
  std::atomic<size_t> misses_{0};
  std::atomic<size_t> set_hits_{0};
  std::unordered_map<Key, Cell, HashFunction> cells_;

  CacheQueue<Key> queue_;
  LoadFunc load_func_;
  // std::mutex mtx_;
  pthread_rwlock_t rw_lock_;

 public:
  LRUCache(size_t max_size, LoadFunc func)
      : max_size_(std::max(static_cast<size_t>(1), max_size)) {
    max_overflow_ = max_size / 20;
    if(max_overflow_ > 1000) {
      max_overflow_ = 1000;
    }
    load_func_ = func;
    LOG(INFO) << "LruCache open! Max_size[" << max_size_ << "], max_overflow["
              << max_overflow_ << "]";
  }

  virtual ~LRUCache() {
    pthread_rwlock_destroy(&rw_lock_);
    LOG(INFO) << "Lrucache destroyed successfully!";
  }

  int Init() {
    queue_.Init();
    int ret = pthread_rwlock_init(&rw_lock_, nullptr);
    if (ret != 0) {
      LOG(ERROR) << "init read-write lock error, ret=" << ret;
      return 2;
    }
    return 0;
  }

  bool Get(Key key, std::shared_ptr<Mapped> &mapped) {
    pthread_rwlock_rdlock(&rw_lock_);
    bool res = GetImpl(key, mapped);
    pthread_rwlock_unlock(&rw_lock_);

    if (res)
      ++hits_;
    else
      ++misses_;
    if (hits_ % 1000000 == 0 && hits_ != last_show_log_) {
      LOG(INFO) << "LruCache cur_size[" << cur_size_ << "] cells_size["
                << cells_.size() << "] hits[" << hits_ << "] set_hits["
                << set_hits_ << "] misses[" << misses_ << "]";
      last_show_log_ = hits_;
    }
    return res;
  }

  void Set(Key key, std::shared_ptr<Mapped> &mapped) {
    // std::lock_guard<std::mutex> lock(mtx_);
    pthread_rwlock_wrlock(&rw_lock_);
    SetImpl(key, mapped);
    pthread_rwlock_unlock(&rw_lock_);
  }

  bool SetOrGet(Key key, std::shared_ptr<Mapped> &load_mapped,
                FuncToken token) {
    std::shared_ptr<InsertInfo> insert_info;

    // std::lock_guard<std::mutex> cache_lck(mtx_);
    pthread_rwlock_wrlock(&rw_lock_);
    bool res = GetImpl2(key, load_mapped);
    if (res) {
      pthread_rwlock_unlock(&rw_lock_);
      ++set_hits_;
      return true;
    }
    auto res_ite = insert_infos_.find(key);
    if (res_ite == insert_infos_.end()) {
      insert_info.reset(new InsertInfo(*this));
      insert_infos_.insert(std::make_pair(key, insert_info));
    } else {
      insert_info = res_ite->second;
    }
    pthread_rwlock_unlock(&rw_lock_);

    InsertInfo *insert = insert_info.get();
    std::lock_guard<std::mutex> insert_lck(insert->mtx_);

    if (insert->is_product_) {
      ++set_hits_;
      load_mapped = insert->value_;
      return true;
    }
    res = load_func_(key, load_mapped, token);
    if (res) {
      insert->value_ = load_mapped;
      insert->is_product_ = true;
    }

    // std::lock_guard<std::mutex> cache_lck(mtx_);
    pthread_rwlock_wrlock(&rw_lock_);
    auto ite = insert_infos_.find(key);
    if (res && ite != insert_infos_.end() && ite->second.get() == insert) {
      SetImpl(key, insert->value_);
    }

    if (!insert_info->is_clean_) {
      insert->is_clean_ = true;
      insert_infos_.erase(key);
    }
    pthread_rwlock_unlock(&rw_lock_);
    return res;
  }

  void Evict(Key key) {
    // std::lock_guard<std::mutex> lock(mtx_);
    pthread_rwlock_wrlock(&rw_lock_);
    auto ite = cells_.find(key);
    if (ite == cells_.end()) {
      pthread_rwlock_unlock(&rw_lock_);
      return;
    }
    auto que_ite = ite->second.queue_ite;
    cells_.erase(ite);
    --cur_size_;

    queue_.Erase(que_ite);
    pthread_rwlock_unlock(&rw_lock_);
  }

  void AlterMaxSize(size_t max_size) {
    max_size_ = max_size;
    max_overflow_ = max_size / 20;
    if(max_overflow_ > 1000) {
      max_overflow_ = 1000;
    }
    LOG(INFO) << "LruCache Max_size[" << max_size_ << "], max_overflow["
              << max_overflow_ << "]";
  }

  size_t GetMaxSize() {
    return max_size_;
  }

  size_t Count() const { return cur_size_; }

  size_t GetHits() { return hits_; }

  size_t GetSetHits() { return set_hits_; }

  size_t GetMisses() { return misses_; }

 private:
  bool GetImpl(const Key &key, std::shared_ptr<Mapped> &mapped) {
    auto ite = cells_.find(key);
    if (ite == cells_.end()) {
      return false;
    }
    Cell &cell = ite->second;
    mapped = cell.value;

    if (cell.hits >= THRESHOLD_OF_SWAP) {
      pthread_rwlock_unlock(&rw_lock_);
      pthread_rwlock_wrlock(&rw_lock_);
      queue_.MoveToTail(cell.queue_ite);
    } else {
      ++cell.hits;
    }
    return true;
  }

  bool GetImpl2(const Key &key, std::shared_ptr<Mapped> &mapped) {
    auto ite = cells_.find(key);
    if (ite == cells_.end()) {
      return false;
    }
    Cell &cell = ite->second;
    mapped = cell.value;

    if (cur_size_ >= max_size_) {
      if (cell.hits >= THRESHOLD_OF_SWAP) {
        queue_.MoveToTail(cell.queue_ite);
      } else {
        ++cell.hits;
      }
    }
    return true;
  }

  void SetImpl(const Key &key, const std::shared_ptr<Mapped> &add_mapped) {
    auto res =
        cells_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                       std::forward_as_tuple());
    Cell &cell = res.first->second;

    bool is_emplace = res.second;
    if (is_emplace) {
      cell.queue_ite = queue_.Insert(key);
      cell.hits = 0;
      ++cur_size_;
      EvictOverflow();
    } else {
      if (cell.hits >= THRESHOLD_OF_SWAP) {
        queue_.MoveToTail(cell.queue_ite);
      } else {
        ++cell.hits;
      }
    }
    cell.value = add_mapped;
  }

  void EvictOverflow() {
    if (cur_size_ >= max_size_ + max_overflow_) {
      int evict_num = cur_size_ - max_size_;
      cur_size_ -= evict_num;

      int fail_pop_num = 0;
      Key key;
      for (int i = 0; i < evict_num; ++i) {
        if (!queue_.Pop(key)) {
          ++fail_pop_num;
          continue;
        }
        auto ite = cells_.find(key);
        if (ite == cells_.end()) {
          LOG(ERROR) << "error, LruCache queue and map is inconsistent.";
          abort();
        }
        cells_.erase(ite);
      }
      cur_size_ += fail_pop_num;
    }
  }
};
