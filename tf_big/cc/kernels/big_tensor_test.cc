#include <string>

#include "tensorflow/core/framework/variant_tensor_data.h"

#include "gtest/gtest.h"
#include "tf_big/cc/kernels/big_tensor.h"

using tf_big::BigTensor;

TEST(BigTensorTest, EncodeDecode) {
  std::string input("2344134134");
  BigTensor b(mpz_class(input, 10));

  tensorflow::VariantTensorData d;

  b.Encode(&d);

  BigTensor b2;

  b2.Decode(d);

  EXPECT_EQ(b2(0, 0).get_str(10), input);
}

// TODO(justin1121) I don't think we need a main function but I couldn't make it
// work without!
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
