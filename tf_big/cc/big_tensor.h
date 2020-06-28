#ifndef TF_BIG_CC_BIG_TENSOR_H_
#define TF_BIG_CC_BIG_TENSOR_H_

#include <gmp.h>
#include <gmpxx.h>

#include <string>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

using Eigen::Dynamic;
using Eigen::Index;
using Eigen::Matrix;

using namespace tensorflow;  // NOLINT

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

namespace tf_big {

inline void encode_length(uint8_t* buffer, unsigned int len) {
  buffer[0] = len & 255;
  buffer[1] = (len >> 8) & 255;
  buffer[2] = (len >> 16) & 255;
  buffer[3] = (len >> 24) & 255;
}
inline unsigned int decode_length(uint8_t* buffer) {
  return buffer[0] + 256 * buffer[1] + 65536 * buffer[2] + 16777216 * buffer[3];
}

struct BigTensor {
  BigTensor() {}
  BigTensor(const BigTensor& other);
  explicit BigTensor(mpz_class m);
  explicit BigTensor(const MatrixXm& mat);

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

    if ((rows == 1) && (cols == 1)) {
      auto mat = t->scalar<T>();
      mat(0) = value(0, 0).get_str();
    } else {
      auto mat = t->matrix<T>();
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          mat(i, j) = value(i, j).get_str();
        }
      }
    }
  }
 template <typename T>
 void LimbsFromTensor(const Tensor& t) {
     int rows = t.dim_size(0);
     int cols = t.dim_size(1);
     int num_bytes_chunks = t.dim_size(2);

     value = MatrixXm(rows, cols);
     for (int i = 0; i < rows; i++) {
       for (int j = 0; j < cols; j++) {
         // copy tensor contents to buffer
         T tmp[num_bytes_chunks];
         for (int k = 0; k < num_bytes_chunks; k++)
             tmp[k] = t.tensor<T, 3>()(i, j, k);

         uint8_t* buffer = reinterpret_cast<uint8_t*>(tmp);

         size_t skip = 0;
         unsigned int length = decode_length(buffer);
         skip += 4;
         // this won't really work if we get int32
         mpz_import(value(i, j).get_mpz_t(), length, 1, sizeof(uint8_t), 0, 0, buffer + skip);
       }
     }
 }

 template <typename T>
 void LimbsToTensor(Tensor* t) const {
     auto rows = value.rows();
     auto cols = value.cols();

     size_t expansion_factor = t->dim_size(2) * sizeof(T);
     uint8_t out_bytes[expansion_factor];

     for (int i = 0; i < rows; i++) {
       for (int j = 0; j < cols; j++) {
         unsigned int num = mpz_sizeinbase(value(i, j).get_mpz_t(), 256);
         size_t skip = 0;
         encode_length(out_bytes, num);
         skip += 4;

         size_t ll;
         mpz_export(out_bytes + skip, &ll, 1, sizeof(uint8_t), 0, 0, value(i, j).get_mpz_t());
         if (ll >= expansion_factor + skip) {
           // how to throw an exception here?
         }
         for (size_t k = ll+skip; k < expansion_factor; k++)
          out_bytes[k] = 0;

         T* tmp = reinterpret_cast<T*>(out_bytes);
         // copy buffer contents to tensor
         for (size_t k = 0; k < expansion_factor/sizeof(T); k++) {
           t->tensor<T, 3>()(i, j, k) = tmp[k];
         }
       }
     }
 }

  BigTensor& operator+=(const BigTensor& rhs) {
    this->value += rhs.value;
    return *this;
  }

  // friend makes this a non-member
  friend BigTensor operator+(BigTensor lhs, const BigTensor& rhs) {
    lhs += rhs;
    return lhs;
  }

  BigTensor& operator-=(const BigTensor& rhs) {
    this->value -= rhs.value;
    return *this;
  }

  // friend makes this a non-member
  friend BigTensor operator-(BigTensor lhs, const BigTensor& rhs) {
    lhs -= rhs;
    return lhs;
  }

  BigTensor& operator*=(const BigTensor& rhs) {
    this->value *= rhs.value;
    return *this;
  }

  // friend makes this a non-member
  friend BigTensor operator*(BigTensor lhs, const BigTensor& rhs) {
    lhs *= rhs;
    return lhs;
  }

  mpz_class operator()(Index i, Index j) const { return value(i, j); }

  BigTensor cwiseProduct(const BigTensor& rhs) const {
    return BigTensor(this->value.cwiseProduct(rhs.value));
  }

  BigTensor cwiseQuotient(const BigTensor& rhs) const {
    return BigTensor(this->value.cwiseQuotient(rhs.value));
  }

  Index rows() const { return value.rows(); }

  Index cols() const { return value.cols(); }

  TensorShape shape() const { return TensorShape{value.rows(), value.cols()}; }

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
inline void BigTensor::ToTensor<uint8>(Tensor* t) const {
    auto rows = value.rows();
    auto cols = value.cols();

    auto mat = t->matrix<uint8>();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = (uint8)value(i, j).get_si();
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


}  // namespace tf_big

#endif  // TF_BIG_CC_BIG_TENSOR_H_
