/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>
#include <unistd.h>
// #include <malloc.h>

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
class CacheList {
 public:
  struct Node {
    Value val;
    Node *prev;
    Node *next;
  };
  CacheList() {
    head_ = nullptr;
    tail_ = nullptr;
  }
  ~CacheList() {
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
    if (!node || node->prev == nullptr || node->next == nullptr || node == tail_) {
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

template <typename Value>
class CacheQueue {
 public:
  struct Node {
    Value val;
    Node *next;
  };

  CacheQueue() {
    head_ = nullptr;
    size_ = 0;
  }

  ~CacheQueue() {
    while (head_) {
      Node *del = head_;
      head_ = del->next;
      delete del;
      del = nullptr;
    }
    tail_ = nullptr;
    size_ = 0;
  }

  void Init() {
    head_ = new Node;
    tail_ = head_;
    head_->next = nullptr;
  }

  void Push(Value value) {
    tail_->next = new Node;
    tail_ = tail_->next;
    tail_->val = value;
    tail_->next = nullptr;
    ++size_;
  }

  bool Pop(Value &value) {
    if (head_ != tail_) {
      Node *del = head_->next;
      value = del->val;
      head_->next = del->next;
      if (del->next == nullptr) {
        tail_ = head_;
      }
      delete del;
      --size_;
      return true;
    }
    return false;
  }

  uint32_t size_;
 private:
  Node *head_;
  Node *tail_;
};

class MemoryPool {
 public:
  MemoryPool(void *(*create_cell_fun)(uint32_t),
             void (*del_cell_fun)(void*)) {
    del_cell_fun_ = del_cell_fun;
    create_cell_fun_ = create_cell_fun;
  }

  ~MemoryPool() {
    ClearQueue();
  }

  void Init(uint32_t max_cell_num, uint32_t cell_size) {
    que_.Init();
    cell_size_ = cell_size;             // uint: byte
    max_cell_num_ = max_cell_num;   
    LOG(INFO) << "MemoryPool info, cell_size_=" << cell_size_ << ",max_cell_num_=" 
              << max_cell_num_ << ",use_cell_num_=" << use_cell_num_;
  }

  void *GetCell() {
    void *cell = nullptr;
    if (que_.size_ > 0) {
      que_.Pop(cell);
    } else {
      cell = create_cell_fun_(cell_size_);
    }
    ++use_cell_num_;
    return cell;
  }

  void ReclaimCell(void *cell) {
    que_.Push(cell);
    --use_cell_num_;
    if (use_cell_num_ + que_.size_ > max_cell_num_) {
      que_.Pop(cell);
      del_cell_fun_(cell);
    }
  }

  uint32_t SetMaxCellNum(uint32_t max_cell_num) {
    if (que_.size_ + use_cell_num_ <= max_cell_num) {
      max_cell_num_ = max_cell_num;
      return max_cell_num_;
    }

    uint32_t del_num = 0;
    if (use_cell_num_ > max_cell_num) {
      del_num = que_.size_;
    } else {
      del_num = que_.size_ + use_cell_num_ - max_cell_num;
    }

    for (uint32_t i = 0; i < del_num; ++i) {
      void *del = nullptr;
      que_.Pop(del);
      del_cell_fun_(del);
      del = nullptr;
    }
    max_cell_num_ -= del_num;
    return max_cell_num;
  }

  uint32_t UseCellNum() {
    return use_cell_num_;
  }

 private:
  void ClearQueue() {
    while (que_.size_ > 0) {
      void *del = nullptr;
      que_.Pop(del);
      del_cell_fun_(del);
      del = nullptr;
    }
  }
  
  uint32_t cell_size_ = 0;
  uint32_t max_cell_num_ = 0;
  uint32_t use_cell_num_ = 0;
  void (*del_cell_fun_)(void*);
  void *(*create_cell_fun_)(uint32_t);

  CacheQueue<void*> que_;
};



template <typename Key, typename FuncToken,
          typename HashFunction = std::hash<Key>>
class LRUCache {
 public:
  using LoadFunc = bool (*)(Key, char *, FuncToken);

  struct Cell {
    char *value = nullptr;
    void *queue_ite = nullptr;
    THRESHOLD_TYPE hits;
    ~Cell() {
      if (value) {
        delete[] value;
        value = nullptr;
        // LOG(INFO) << "~Cell";
      }
    }

    static void *CreateCell(uint32_t val_len) {
      Cell *cell = new Cell;
      cell->value = new char[val_len];
      return (void*)cell;
    }

    static void DeleteCell(void *cell) {
      if (cell) {
        Cell *del = (Cell*)cell;
        delete del;
        cell = nullptr;
        del = nullptr;
      }
    }
  };

  struct InsertInfo {
    std::mutex mtx_;
    bool is_clean_ = false;
    bool is_product_ = false;
    // std::shared_ptr<Mapped> value_;
    Cell *cell_ = nullptr;
  };


 private:
  std::string name_;
  size_t max_size_;
  size_t cell_size_;
  MemoryPool mem_pool_;
  std::unordered_map<Key, std::shared_ptr<InsertInfo>, HashFunction>
      insert_infos_;
  size_t max_overflow_ = 0;
  size_t last_show_log_ = 1;
  std::atomic<size_t> cur_size_{0};
  std::atomic<size_t> hits_{0};
  std::atomic<size_t> misses_{0};
  std::atomic<size_t> set_hits_{0};
  std::unordered_map<Key, Cell *, HashFunction> cells_;

  CacheList<Key> queue_;
  LoadFunc load_func_;
  std::mutex mtx_;
  // pthread_rwlock_t rw_lock_;

 public:
  LRUCache(std::string name, size_t cache_size, size_t cell_size,
           LoadFunc func) : mem_pool_(&Cell::CreateCell, &Cell::DeleteCell){
    name_ = name;
    cell_size_ = cell_size;
    max_size_ = (cache_size * 1024 * 1024) / cell_size; 
    max_overflow_ = max_size_ / 20;
    if(max_overflow_ > 1000) {
      max_overflow_ = 1000;
    }
    max_size_ -= max_overflow_;
    load_func_ = func;
    LOG(INFO) << "LruCache[" << name_ << "] open! Max_size[" << max_size_ 
              << "], max_overflow[" << max_overflow_ << "]";
  }

  virtual ~LRUCache() {
    // pthread_rwlock_destroy(&rw_lock_);
    LOG(INFO) << "LruCache[" << name_ << "] destroyed successfully!";
  }

  int Init() {
    queue_.Init();
    mem_pool_.Init((uint32_t)max_size_ + max_overflow_ + 500, (uint32_t)cell_size_);
    // int ret = pthread_rwlock_init(&rw_lock_, nullptr);
    // if (ret != 0) {
    //   LOG(ERROR) << "LruCache[" << name_ 
    //              << "] init read-write lock error, ret=" << ret;
    //   return 2;
    // }
    return 0;
  }

  bool Get(Key key, char *&mapped) {
    bool res;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      // pthread_rwlock_rdlock(&rw_lock_);
      res = GetImpl(key, mapped);
      // pthread_rwlock_unlock(&rw_lock_);
    }

    if (res)
      ++hits_;
    else
      ++misses_;
    if (hits_ % 100000 == 0 && hits_ != last_show_log_) {
      LOG(INFO) << "LruCache[" << name_ << "] cur_size[" << cur_size_
                << "] cells_size[" << cells_.size() << "] hits[" << hits_
                << "] set_hits[" << set_hits_ << "] misses[" << misses_ << "]";
      last_show_log_ = hits_;
    }
    return res;
  }

  void Set(Key key, char *mapped) {
    std::lock_guard<std::mutex> lock(mtx_);
    // pthread_rwlock_wrlock(&rw_lock_);
    SetImpl(key, mapped);
    // pthread_rwlock_unlock(&rw_lock_);
  }

  bool SetOrGet(Key key, char *&load_mapped,
                FuncToken token) {
    std::shared_ptr<InsertInfo> ptr_insert;
    {
      std::lock_guard<std::mutex> cache_lck(mtx_);
      // pthread_rwlock_wrlock(&rw_lock_);
      bool res = GetImpl(key, load_mapped);
      if (res) {
        // pthread_rwlock_unlock(&rw_lock_);
        ++set_hits_;
        return true;
      }

      auto &insert_info = insert_infos_[key];
      if (!insert_info) {
        insert_info = std::make_shared<InsertInfo>();
        insert_info->cell_ = (Cell*)(mem_pool_.GetCell());
      }
      ptr_insert = insert_info;
    // pthread_rwlock_unlock(&rw_lock_);
    }

    InsertInfo *insert = ptr_insert.get();
    std::lock_guard<std::mutex> insert_lck(insert->mtx_);

    if (insert->is_product_) {
      ++set_hits_;
      load_mapped = insert->cell_->value;
      return true;
    }
    bool res = load_func_(key, insert->cell_->value, token);
    if (res) {
      load_mapped = insert->cell_->value;
      insert->is_product_ = true;
    }

    std::lock_guard<std::mutex> cache_lck(mtx_);
    // pthread_rwlock_wrlock(&rw_lock_);
    auto ite = insert_infos_.find(key);
    if (res && ite != insert_infos_.end() && ite->second.get() == insert) {
      SetImpl(key, insert->cell_);
    } else {
      mem_pool_.ReclaimCell((void*)(insert->cell_));
    }

    if (!ptr_insert->is_clean_) {
      insert->is_clean_ = true;
      insert_infos_.erase(key);
    }
    // pthread_rwlock_unlock(&rw_lock_);
    return res;
  }

  void Evict(Key key) {
    std::lock_guard<std::mutex> lock(mtx_);
    // pthread_rwlock_wrlock(&rw_lock_);
    auto ite = cells_.find(key);
    if (ite == cells_.end()) {
      // pthread_rwlock_unlock(&rw_lock_);
      return;
    }
    Cell *del = ite->second;
    auto que_ite = del->queue_ite;
    mem_pool_.ReclaimCell((void*)del);
    cells_.erase(ite);
    --cur_size_;

    queue_.Erase(que_ite);
    // pthread_rwlock_unlock(&rw_lock_);
  }

  void AlterCacheSize(size_t cache_size) {
    max_size_ = (cache_size * 1024 * 1024) / cell_size_; 
    max_overflow_ = max_size_ / 20;
    if(max_overflow_ > 1000) {
      max_overflow_ = 1000;
    }
    max_size_ = max_size_ - max_overflow_;
    // pthread_rwlock_wrlock(&rw_lock_);
    std::lock_guard<std::mutex> lock(mtx_);
    EvictOverflow();
    mem_pool_.SetMaxCellNum((uint32_t)max_size_ + 500);
    // pthread_rwlock_unlock(&rw_lock_);
    LOG(INFO) << "LruCache[" << name_ << "] Max_size[" << max_size_ 
              << "], max_overflow[" << max_overflow_ << "]";
  }

  size_t GetMaxSize() {
    return max_size_ + max_overflow_;
  }

  size_t Count() const { return cur_size_; }

  size_t GetHits() { return hits_; }

  size_t GetSetHits() { return set_hits_; }

  size_t GetMisses() { return misses_; }

  std::string GetName() { return name_; }

 private:
  
  bool GetImpl(const Key &key, char *&mapped) {
    auto ite = cells_.find(key);
    if (ite == cells_.end()) {
      return false;
    }
    Cell *&cell = ite->second;
    mapped = cell->value;

    if (cell->hits >= THRESHOLD_OF_SWAP) {
      // pthread_rwlock_unlock(&rw_lock_);
      // pthread_rwlock_wrlock(&rw_lock_);
      queue_.MoveToTail(cell->queue_ite);
      cell->hits = 0;
    } else {
      ++(cell->hits);
    }
    return true;
  }

  
  void SetImpl(const Key &key, Cell *add_cell) {
    auto res =
        cells_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                       std::forward_as_tuple());
    Cell* &cell = res.first->second;
    bool inserted = res.second;

    if (inserted) {
      cell = add_cell;
      cell->queue_ite = queue_.Insert(key);
      cell->hits = 0;
      ++cur_size_;
      EvictOverflow();
    } else {
      if (cell->hits >= THRESHOLD_OF_SWAP) {
        queue_.MoveToTail(cell->queue_ite);
        add_cell->hits = 0;
      } else {
        add_cell->hits = cell->hits;
        ++add_cell->hits;
      }
      add_cell->queue_ite = cell->queue_ite;
      mem_pool_.ReclaimCell((void*)cell);
      cell = add_cell;
    }
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
          LOG(ERROR) << "Lrucache[" << name_ << "] queue_.Pop(" << key <<") failed.";
          continue;
        }
        auto ite = cells_.find(key);
        if (ite == cells_.end()) {
          LOG(ERROR) << "LruCache[" << name_ << "], cur_size[" << cur_size_ 
                     << "], cells_.size()[" << cells_.size() << "]."
                     << "Queue and map is inconsistent.";
          continue;
          // abort();
        }
        mem_pool_.ReclaimCell((void*)(ite->second));
        cells_.erase(ite);
      }
      cur_size_ += fail_pop_num;
      // malloc_trim(0);
    }
  }
};
