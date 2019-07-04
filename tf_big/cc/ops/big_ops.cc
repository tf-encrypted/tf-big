#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("BigImport")
    .Attr("dtype: {int32, string}")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful();

REGISTER_OP("BigExport")
    .Attr("dtype: {int32, string}")
    .Input("val: variant")
    .Output("out: dtype")
    .SetIsStateful();

REGISTER_OP("BigAdd")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_OP("BigSub")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_OP("BigMul")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_OP("BigMatMul")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();
