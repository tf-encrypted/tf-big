#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "big_tensor.h"
#include "gmp.h"

using namespace tensorflow;

Status GetBigTensor(OpKernelContext* ctx, int index, const BigTensor** res) {
  const Tensor& input = ctx->input(index);

  // TODO: check scalar type
  const BigTensor* big = input.scalar<Variant>()().get<BigTensor>();
  if (big == nullptr) {
    return errors::InvalidArgument("Input handle is not a big tensor. Saw: '",
                                   input.scalar<Variant>()().DebugString(),
                                   "'");
  }

  *res = big;
  return Status::OK();
}

class BigImportOp : public OpKernel {
 public:
  explicit BigImportOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& str = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(str.shape()),
        errors::InvalidArgument("value expected to be a matrix ",
                                "but got shape: ", str.shape().DebugString()));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    BigTensor big;
    big.FromTensor<string>(str);

    val->scalar<Variant>()() = std::move(big);
  }
};

class BigExportOp : public OpKernel {
 public:
  explicit BigExportOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const BigTensor* val = nullptr;
    OP_REQUIRES_OK(ctx, GetBigTensor(ctx, 0, &val));
    auto output_shape = TensorShape{val->rows(), val->cols()};

    Tensor* str;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &str));

    val->ToTensor<string>(str);
  }
};

class BigAddOp : public OpKernel {
 public:
  explicit BigAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const BigTensor* val1 = nullptr;
    OP_REQUIRES_OK(ctx, GetBigTensor(ctx, 0, &val1));

    const BigTensor* val2 = nullptr;
    OP_REQUIRES_OK(ctx, GetBigTensor(ctx, 1, &val2));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));

    auto res = *val1 + *val2;

    // TODO: free old memory???
    output->scalar<Variant>()() = std::move(res);
  }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(BigTensor, BigTensor::kTypeName);

REGISTER_KERNEL_BUILDER(Name("BigImport").Device(DEVICE_CPU), BigImportOp);

REGISTER_KERNEL_BUILDER(Name("BigExport").Device(DEVICE_CPU), BigExportOp);

REGISTER_KERNEL_BUILDER(Name("BigAdd").Device(DEVICE_CPU), BigAddOp);
