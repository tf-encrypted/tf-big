#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Input just a string for now, we make this more robust in the future
REGISTER_OP("CreateMpzVariant")
    .Input("value: string")
    .Output("mpz: variant")
    .SetIsStateful();

REGISTER_OP("AddMpz")
    .Input("val1: variant")
    .Input("val2: variant")
    .Output("res: variant")
    .SetIsStateful();


REGISTER_OP("MpzToString")
    .Input("mpz: variant")
    .Output("str: string")
    .SetIsStateful();