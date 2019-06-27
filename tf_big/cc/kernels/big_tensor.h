#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "gmp.h"
#include "gmpxx.h"

using Eigen::Dynamic;
using Eigen::Matrix;

using namespace tensorflow;

namespace Eigen {
template <>
struct NumTraits<mpz_class> : GenericNumTraits<mpz_class> {
  typedef mpz_class Real;
  typedef mpz_class NonInteger;
  typedef mpz_class Nested;
  static inline Real epsilon() { return 0; }
  static inline Real dummy_precision() { return 0; }
  static inline int digits10() { return 0; }

  enum {
    IsInteger = 0,
    IsSigned = 1,
    IsComplex = 0,
    RequireInitialization = 1,
    ReadCost = 6,
    AddCost = 150,
    MulCost = 100
  };
};
}  // namespace Eigen

typedef Matrix<mpz_class, Dynamic, Dynamic> MatrixXm;

namespace tfbig {

struct BigTensor {
 public:
  BigTensor(){};
  BigTensor(const BigTensor& other);
  BigTensor(mpz_class m);
  BigTensor(const MatrixXm& mat);

  static const char kTypeName[];
  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  string DebugString() const { return "BigTensor"; }

  template <typename T>
  void FromTensor(const Tensor& t) {
    auto rows = t.dim_size(0);
    auto cols = t.dim_size(1);

    value = MatrixXm(rows, cols);

    auto mat = t.matrix<T>();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        value(i, j) = mpz_class(mat(i, j));
      }
    }
  }


  template <typename T>
  void ToTensor(Tensor* t) const {
    auto rows = value.rows();
    auto cols = value.cols();

    auto mat = t->matrix<T>();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat(i, j) = value(i, j).get_str();
      }
    }
  }

  MatrixXm value;
};

template <>
inline void BigTensor::ToTensor<int32>(Tensor* t) const {
  auto rows = value.rows();
  auto cols = value.cols();

  auto mat = t->matrix<int32>();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat(i, j) = value(i, j).get_si();
    }
  }
}

template <>
inline void BigTensor::FromTensor<string>(const Tensor& t) {
  auto rows = t.dim_size(0);
  auto cols = t.dim_size(1);

  value = MatrixXm(rows, cols);

  auto mat = t.matrix<string>();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      value(i, j) = mpz_class(mat(i, j), 10);
    }
  }
}

}  // namespace tfbig
