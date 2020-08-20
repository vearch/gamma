/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This faiss source code is licensed under the MIT license.
 * https://github.com/facebookresearch/faiss/blob/master/LICENSE
 *
 *
 * The works below are modified based on faiss:
 * 1. Replace the static batch indexing with real time indexing
 * 2. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2019 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#include "gamma_hnsw.h"

namespace tig_gamma {

GammaHNSW::GammaHNSW(int M):HNSW(M)
{
  nlinks = M;
}

namespace {


using storage_idx_t = faiss::HNSW::storage_idx_t;
using NodeDistCloser = faiss::HNSW::NodeDistCloser;
using NodeDistFarther = faiss::HNSW::NodeDistFarther;
using HNSW = faiss::HNSW;

/**************************************************************
 * Addition subroutines
 **************************************************************/


/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& resultSet1,
  size_t max_size)
{
  if (resultSet1.size() < max_size) {
      return;
  }
  std::priority_queue<NodeDistFarther> resultSet;
  std::vector<NodeDistFarther> returnlist;

  while (resultSet1.size() > 0) {
      resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
      resultSet1.pop();
  }

  faiss::HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);

  for (NodeDistFarther curen2 : returnlist) {
      resultSet1.emplace(curen2.d, curen2.id);
  }
}


/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(GammaHNSW& hnsw,
              DistanceComputer& qdis,
              storage_idx_t src, storage_idx_t dest,
              int level)
{
  size_t begin, end;
  hnsw.neighbor_range(src, level, &begin, &end);
  if (hnsw.neighbors[end - 1] == -1) {
    // there is enough room, find a slot to add it
    size_t i = end;
    while(i > begin) {
      if (hnsw.neighbors[i - 1] != -1) break;
      i--;
    }
    hnsw.neighbors[i] = dest;
    return;
  }

  // otherwise we let them fight out which to keep

  // copy to resultSet...
  std::priority_queue<NodeDistCloser> resultSet;
  resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
  for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
    storage_idx_t neigh = hnsw.neighbors[i];
    resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
  }

  shrink_neighbor_list(qdis, resultSet, end - begin);

  // ...and back
  size_t i = begin;
  while (resultSet.size()) {
    hnsw.neighbors[i++] = resultSet.top().id;
    resultSet.pop();
  }
  // they may have shrunk more than just by 1 element
  while(i < end) {
    hnsw.neighbors[i++] = -1;
  }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
  GammaHNSW& hnsw,
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& results,
  int entry_point,
  float d_entry_point,
  int level)
{
  std::set<int> visited_node;
  // top is nearest candidate
  std::priority_queue<NodeDistFarther> candidates;

  NodeDistFarther ev(d_entry_point, entry_point);
  candidates.push(ev);
  results.emplace(d_entry_point, entry_point);
  visited_node.insert(entry_point);

  while (!candidates.empty()) {
    // get nearest
    const NodeDistFarther &currEv = candidates.top();

    if (currEv.d > results.top().d) {
      break;
    }
    int currNode = currEv.id;
    candidates.pop();

    // loop over neighbors
    size_t begin, end;
    hnsw.neighbor_range(currNode, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t nodeId = hnsw.neighbors[i];
      if (nodeId < 0) break;
      if (visited_node.count(nodeId)) continue;
      visited_node.insert(nodeId);

      float dis = qdis(nodeId);
      NodeDistFarther evE1(dis, nodeId);

      if (results.size() < (size_t)hnsw.efConstruction ||
          results.top().d > dis) {

        results.emplace(dis, nodeId);
        candidates.emplace(dis, nodeId);
        if (results.size() > (size_t)hnsw.efConstruction) {
          results.pop();
        }
      }
    }
  }
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
void greedy_update_nearest(const GammaHNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t v = hnsw.neighbors[i];
      if (v < 0) break;
      float dis = qdis(v);
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    if (nearest == prev_nearest) {
      return;
    }
  }
}

}  // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void GammaHNSW::AddLinksStartingFrom(DistanceComputer& ptdis,
                                   storage_idx_t pt_id,
                                   storage_idx_t nearest,
                                   float d_nearest,
                                   int level,
                                   omp_lock_t *locks)
{
  std::priority_queue<NodeDistCloser> link_targets;

  search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest,
                          level);

  // but we can afford only this many neighbors
  int M = nb_neighbors(level);

  tig_gamma::shrink_neighbor_list(ptdis, link_targets, M);

  while (!link_targets.empty()) {
    int other_id = link_targets.top().id;

    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level);
    omp_unset_lock(&locks[other_id]);

    add_link(*this, ptdis, pt_id, other_id, level);

    link_targets.pop();
  }
}

int GammaHNSW::AddWithLocks(DistanceComputer& ptdis, int pt_level, 
                          int pt_id, std::vector<omp_lock_t>& locks)
{
  //  greedy search on upper levels
  storage_idx_t nearest;
#pragma omp critical
  {
    nearest = entry_point;

    if (nearest == -1) {
      max_level = pt_level;
      entry_point = pt_id;
    }
  }

  if (nearest < 0) {
    return -1;
  }

  omp_set_lock(&locks[pt_id]);

  int level = max_level; // level at which we start adding neighbors
  float d_nearest = ptdis(nearest);

  for(; level > pt_level; level--) {
    greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
  }
  
  for(; level >= 0; level--) {
    AddLinksStartingFrom(ptdis, pt_id, nearest, d_nearest,
                          level, locks.data());
  }

  omp_unset_lock(&locks[pt_id]);

  if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
  }
  return 0;
}

/** Do a BFS on the candidates list */

int GammaHNSW::SearchFromCandidates(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  int level, const char * docids_bitmap,
  MultiRangeQueryResults *range_query_result,
  int nres_in) const
{
  std::set<int> visited_node;

  int nres = nres_in;
  int ndis = 0;
  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    visited_node.insert(v1);

    if (bitmap::test(docids_bitmap, v1) ||
         (range_query_result &&
          not range_query_result->Has(v1))) {
      continue;
    }

    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
  }

  bool do_dis_check = check_relative_distance;
  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    if (do_dis_check) {
      // tricky stopping condition: there are more that ef
      // distances that are processed already that are smaller
      // than d0

      int n_dis_below = candidates.count_below(d0);
      if(n_dis_below >= efSearch) {
        break;
      }
    }

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (bitmap::test(docids_bitmap, v1) ||
          (range_query_result && 
            not range_query_result->Has(v1))) {
        continue;
      }
      if (visited_node.count(v1)) {
        continue;
      }
      visited_node.insert(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    if (!do_dis_check && nstep > efSearch) {
      break;
    }
  }

  return nres;
}

/**************************************************************
 * Searching
 **************************************************************/

std::priority_queue<Node> GammaHNSW::SearchFromCandidateUnbounded(
  const Node& node,
  DistanceComputer& qdis,
  size_t ef, const char * docids_bitmap,
  MultiRangeQueryResults *range_query_result) const
{
  int ndis = 0;
  std::priority_queue<Node> top_candidates;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
  if (bitmap::test(docids_bitmap, node.second) == false) {
    if ((range_query_result == nullptr) ||
        range_query_result->Has(node.second))
    top_candidates.push(node);
  } 

  std::set<int> visited_node;

  visited_node.insert(node.second);

  while (!candidates.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidates.top();

    if (top_candidates.size() > 0 && d0 > top_candidates.top().first) {
      break;
    }

    candidates.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) break;
      if (visited_node.count(v1)) {
        continue;
      }
      visited_node.insert(v1);

      if (bitmap::test(docids_bitmap, v1) ||
          (range_query_result && 
            not range_query_result->Has(v1))) {
        continue;
      }

      float d1 = qdis(v1);
      ++ndis;

      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidates.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
      }
    }
  }

  return top_candidates;
}

void GammaHNSW::Search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  const char * docids_bitmap,
                  MultiRangeQueryResults *range_query_result) const
{
  if (entry_point == -1)
  {
    LOG(ERROR) << "Index is empty, need to build index first.";
    return;
  }

  //  greedy search on upper levels
  storage_idx_t nearest = entry_point;
  float d_nearest = qdis(nearest);

  for(int level = max_level; level >= 1; level--) {
    greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
  }
  
  int ef = std::max(efSearch, k);

  if (search_bounded_queue) {
    MinimaxHeap candidates(ef);

    candidates.push(nearest, d_nearest);

    SearchFromCandidates(qdis, k, I, D, candidates,
      0, docids_bitmap, range_query_result);
  } else {
    std::priority_queue<Node> top_candidates =
      SearchFromCandidateUnbounded(Node(d_nearest, nearest),
        qdis, ef,docids_bitmap, range_query_result);

    while (top_candidates.size() > (size_t)k) {
      top_candidates.pop();
    }

    int nres = 0;
    while (!top_candidates.empty()) {
      float d;
      storage_idx_t label;
      std::tie(d, label) = top_candidates.top();
      faiss::maxheap_push(++nres, D, I, d, label);
      top_candidates.pop();
    }
  }
}

}
