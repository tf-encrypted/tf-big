#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

#include "include/gmp.h"

using namespace tensorflow;

struct BigTensor {
 public:
  BigTensor();
  //BigTensor(string value);
  BigTensor(const BigTensor& other);

  static const char kTypeName[];
  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  string DebugString() const { return "BigTensor"; }

  mpz_t value;
};

BigTensor::BigTensor() {
    mpz_init(this->value);
}

BigTensor::BigTensor(const BigTensor& other) {
    mpz_init_set(this->value, other.value);
}

void BigTensor::Encode(VariantTensorData* data) const {
    size_t count_p;

    gmp_printf("HELELLELELELLE %Zd\n", value);

    char * p = (char *)mpz_export(NULL, &count_p, 1, sizeof(unsigned long), 0, 0, value);

    int total_size = count_p * sizeof(unsigned long);

    std::string s(p, total_size);

    data->set_type_name(TypeName());
    data->set_metadata(s);
}

bool BigTensor::Decode(const VariantTensorData& data) {
    string metadata("");
    data.get_metadata(&metadata);
    mpz_import(value, 1, 1, sizeof(unsigned long), 0, 0, metadata.c_str());

    gmp_printf("%Zd", value);

    std::cout << "HI" << std::endl;

    return true;
}

const char BigTensor::kTypeName[] = "BigTensor";

Status GetBigTensor(OpKernelContext* ctx, int index, const BigTensor** res) {
    const Tensor& input = ctx->input(index);

    // TODO: check scalar type
    const BigTensor* big = input.scalar<Variant>()().get<BigTensor>();
    if(big == nullptr) {
        return errors::InvalidArgument(
        "Input handle is not a mpz wrapper. Saw: '",
        input.scalar<Variant>()().DebugString(), "'");
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
        ctx, TensorShapeUtils::IsScalar(str.shape()),
        errors::InvalidArgument(
            "value expected to be a scalar ",
            "but got shape: ", str.shape().DebugString()));

    Tensor* val;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &val));

    BigTensor big;
    mpz_init_set_str(big.value, str.scalar<string>()().c_str(), 10);

    val->scalar<Variant>()() = std::move(big);
  }
};

class BigExportOp : public OpKernel {
public:
    explicit BigExportOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
        const BigTensor* val = nullptr;
        OP_REQUIRES_OK(ctx, GetBigTensor(ctx, 0, &val));

        Tensor* str;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &str));

        char buf[50];
        gmp_sprintf(buf, "%Zd", val->value);

        string s(buf);

        str->scalar<string>()() = s;
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

    Tensor* res;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &res));

    BigTensor big;
    mpz_init(big.value);
    mpz_add(big.value, val1->value, val2->value);

    // TODO: free old memory???
    res->scalar<Variant>()() = std::move(big);
  }
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(BigTensor, BigTensor::kTypeName);

REGISTER_KERNEL_BUILDER(
  Name("BigImport")
  .Device(DEVICE_CPU),
  BigImportOp);

REGISTER_KERNEL_BUILDER(
  Name("BigExport")
  .Device(DEVICE_CPU),
  BigExportOp);

REGISTER_KERNEL_BUILDER(
  Name("BigAdd")
  .Device(DEVICE_CPU),
  BigAddOp);
