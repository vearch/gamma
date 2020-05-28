#include "realtime_mem_data.h"
#include "test.h"
#include "utils.h"

using namespace std;
using namespace tig_gamma;

namespace Test {

class RealTimeMemDataTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { random_generator = new RandomGenerator(); }

  static void TearDownTestSuite() {
    if (random_generator) delete random_generator;
  }

  // You can define per-test set-up logic as usual.
  virtual void SetUp() {
    buckets_num = 3;
    max_vec_size = 10000;
    int bytes_count = 0;
    bitmap::create(docids_bitmap, bytes_count, max_vec_size);
    vid_mgr = new VIDMgr(false);
    bucket_keys = 100;
    bucket_keys_limit = bucket_keys * 10;
    code_byte_size = 64;
    realtime_data = new realtime::RealTimeMemData(
        buckets_num, max_vec_size, vid_mgr, docids_bitmap, bucket_keys,
        bucket_keys_limit, code_byte_size);
    ASSERT_EQ(true, realtime_data->Init());
  }

  // You can define per-test tear-down logic as usual.
  virtual void TearDown() {
    CHECK_DELETE(vid_mgr);
    CHECK_DELETE(realtime_data);
    // CHECK_DELETE_ARRAY(vid2docid);
    CHECK_DELETE_ARRAY(docids_bitmap);
  }

  const uint8_t *GetCode(int bucket_no, int pos) {
    return realtime_data->cur_invert_ptr_->codes_array_[bucket_no] +
           pos * code_byte_size;
  }

  long GetVid(int bucket_no, int pos) {
    return realtime_data->cur_invert_ptr_->idx_array_[bucket_no][pos];
  }

  int GetRetrievePos(int bucket_no) {
    return realtime_data->cur_invert_ptr_->retrieve_idx_pos_[bucket_no];
  }

  int GetBucketKeys(int bucket_no) {
    return realtime_data->cur_invert_ptr_->cur_bucket_keys_[bucket_no];
  }

  int GetDeletedNum(int bucket_no) {
    return realtime_data->cur_invert_ptr_->deleted_nums_[bucket_no];
  }

  long GetTotalCompactedNum() {
    return realtime_data->cur_invert_ptr_->compacted_num_;
  }

  // member
  int bucket_keys;
  int bucket_keys_limit;
  int code_byte_size;
  // int *vid2docid;
  VIDMgr *vid_mgr;
  char *docids_bitmap;
  int max_vec_size;
  int buckets_num;
  realtime::RealTimeMemData *realtime_data;

  // Some expensive resource shared by all tests.
  // static T* shared_resource_;
  static RandomGenerator *random_generator;
};

RandomGenerator *RealTimeMemDataTest::random_generator = nullptr;

TEST_F(RealTimeMemDataTest, CompactBucket) {
  int num = 100, bucket_no = 0;
  std::vector<long> keys;
  std::vector<uint8_t> codes;
  keys.resize(num, -1);
  codes.resize(num * code_byte_size, 0);
  for (int i = 0; i < num; i++) {
    keys[i] = i;
    codes[i * code_byte_size] = i;
  }
  ASSERT_TRUE(realtime_data->AddKeys(bucket_no, num, keys, codes));
  ASSERT_EQ(num, realtime_data->cur_invert_ptr_->retrieve_idx_pos_[bucket_no]);
  ASSERT_EQ(num - 1, GetVid(bucket_no, num - 1));
  ASSERT_EQ(num - 1, GetCode(bucket_no, num - 1)[0]);
  vector<int> deleted_vids;
  for (int i = 0; i < 10; i++) {
    int pos = random_generator->Rand(100);
    if (!bitmap::test(docids_bitmap, pos)) {
      bitmap::set(docids_bitmap, pos);
      deleted_vids.push_back(pos);
    }
  }
  LOG(INFO) << "deleted vids="
            << utils::join(deleted_vids.data(), deleted_vids.size(), ',');
  realtime_data->Delete(deleted_vids.data(), deleted_vids.size() - 1);
  ASSERT_EQ(deleted_vids.size() - 1, GetDeletedNum(bucket_no));
  realtime_data->CompactBucket(bucket_no);
  ASSERT_EQ(num - deleted_vids.size(), GetRetrievePos(bucket_no));
  ASSERT_EQ(bucket_keys, GetBucketKeys(bucket_no));
  ASSERT_EQ(0, GetDeletedNum(bucket_no));
  ASSERT_EQ(deleted_vids.size(), GetTotalCompactedNum());
  int idx = 0;
  for (int i = 0; i < num; i++) {
    if (bitmap::test(docids_bitmap, i)) continue;
    ASSERT_EQ(i, GetVid(bucket_no, idx));
    ASSERT_EQ(i, GetCode(bucket_no, idx++)[0]);
  }
}

void CreateData(int num, std::vector<long> &keys, std::vector<uint8_t> &codes,
                int code_byte_size) {
  keys.resize(num, -1);
  codes.resize(num * code_byte_size, 0);
  for (int i = 0; i < num; i++) {
    keys[i] = i;
    codes[i * code_byte_size] = i;
  }
}

TEST_F(RealTimeMemDataTest, ExtendBucket) {
  int num = bucket_keys, bucket_no = 0;
  std::vector<long> keys;
  std::vector<uint8_t> codes;
  CreateData(num, keys, codes, code_byte_size);

  realtime::RTInvertBucketData *old_invert_prt = realtime_data->cur_invert_ptr_;
  ASSERT_TRUE(realtime_data->AddKeys(bucket_no, num, keys, codes));

  std::vector<long> keys1;
  std::vector<uint8_t> codes1;
  int num1 = 1;
  CreateData(num1, keys1, codes1, code_byte_size);
  ASSERT_TRUE(realtime_data->AddKeys(bucket_no, num1, keys1, codes1));

  ASSERT_NE(old_invert_prt, realtime_data->cur_invert_ptr_);
  ASSERT_EQ(num + num1,
            realtime_data->cur_invert_ptr_->retrieve_idx_pos_[bucket_no]);
  ASSERT_EQ(bucket_keys * 2,
            realtime_data->cur_invert_ptr_->cur_bucket_keys_[bucket_no]);
  ASSERT_EQ(0, GetVid(bucket_no, 0));
  ASSERT_EQ(0, GetCode(bucket_no, 0)[0]);
  ASSERT_EQ(num - 1, GetVid(bucket_no, num - 1));
  ASSERT_EQ(num - 1, GetCode(bucket_no, num - 1)[0]);
}
}  // namespace Test
