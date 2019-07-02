#include "gmp.h"

#include "big_tensor.h"

namespace tf_big {
BigTensor::BigTensor(const MatrixXm& mat) { value = mat; }

BigTensor::BigTensor(const BigTensor& other) { value = other.value; }

BigTensor::BigTensor(mpz_class m) {
  value = MatrixXm(1, 1);
  value(0, 0) = m;
}

void BigTensor::Encode(VariantTensorData* data) const {
  auto rows = value.rows();
  auto cols = value.cols();

  auto shape = TensorShape{rows, cols};
  Tensor t(DT_STRING, shape);

  auto mat = t.matrix<string>();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      size_t count_p;

      char* p = (char*)mpz_export(NULL, &count_p, 1, sizeof(signed long), 0, 0,
                                  value(i, j).get_mpz_t());

      int total_size = count_p * sizeof(signed long);

      mat(i, j) = string(p, total_size);
    }
  }

  *data->add_tensors() = t;

  data->set_type_name(TypeName());
}

bool BigTensor::Decode(const VariantTensorData& data) {
  if(!TensorShapeUtils::IsMatrix(data.tensors()[0].shape())) {
    return false;
  }

  auto mat = data.tensors()[0].matrix<string>();

  auto rows = data.tensors()[0].dim_size(0);
  auto cols = data.tensors()[0].dim_size(1);

  value = MatrixXm(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mpz_import(value(i, j).get_mpz_t(), 1, 1, sizeof(signed long), 0, 0,
                 mat(i, j).c_str());
    }
  }

  return true;
}

const char BigTensor::kTypeName[] = "BigTensor";

}

