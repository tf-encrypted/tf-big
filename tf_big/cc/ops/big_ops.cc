#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Input just a string for now, we make this more robust in the future
REGISTER_OP("BigImport")
    .Input("in: string")
    .Output("val: variant")
    .SetIsStateful();

REGISTER_OP("BigExport")
    .Input("val: variant")
    .Output("out: string")
    .SetIsStateful();

REGISTER_OP("BigAdd")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();
