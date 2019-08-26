#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("BigImport")
    .Attr("dtype: {int32, string}")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 2, &output));
        c->set_output(0, output);
        return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigExport")
    .Attr("dtype: {int32, string}")
    .Input("val: variant")
    .Output("out: dtype")
    .SetIsStateful()
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("BigAdd")
    .Input("val0: variant")
    .Input("val1: variant")
    .Output("res: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle val0 = c->input(0);
        ::tensorflow::shape_inference::ShapeHandle val1 = c->input(1);
        ::tensorflow::shape_inference::ShapeHandle res;
        TF_RETURN_IF_ERROR(c->Merge(val0, val1, &res));
        c->set_output(0, res);
        return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigSub")
    .Input("val0: variant")
    .Input("val1: variant")
    .Output("res: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle val0 = c->input(0);
        ::tensorflow::shape_inference::ShapeHandle val1 = c->input(1);
        ::tensorflow::shape_inference::ShapeHandle res;
        TF_RETURN_IF_ERROR(c->Merge(val0, val1, &res));
        c->set_output(0, res);
        return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigMul")
    .Input("val0: variant")
    .Input("val1: variant")
    .Output("res: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle val0 = c->input(0);
        ::tensorflow::shape_inference::ShapeHandle val1 = c->input(1);
        ::tensorflow::shape_inference::ShapeHandle res;
        TF_RETURN_IF_ERROR(c->Merge(val0, val1, &res));
        c->set_output(0, res);
        return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigPow")
    .Attr("secure: bool")
    .Input("base: variant")
    .Input("exponent: variant")
    .Input("modulus: variant")
    .Output("res: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle base = c->input(0);
        // ::tensorflow::shape_inference::ShapeHandle exponent = c->input(1);
        // ::tensorflow::shape_inference::ShapeHandle modulus = c->input(2);
        // ::tensorflow::shape_inference::ShapeHandle res;
        // TODO(Morten) make sure shapes match
        c->set_output(0, base);
        return ::tensorflow::Status::OK();
    });

// TODO(Morten) add shape inference function
REGISTER_OP("BigMatMul")
    .Input("val0: variant")
    .Input("val1: variant")
    .Output("res: variant")
    .SetIsStateful();

REGISTER_OP("BigMod")
    .Input("val: variant")
    .Input("mod: variant")
    .Output("res: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle val = c->input(0);
        ::tensorflow::shape_inference::ShapeHandle mod = c->input(1);
        TF_RETURN_IF_ERROR(c->WithRankAtMost(val, 2, &val));
        // TODO(Morten) `mod` below should be a scalar
        TF_RETURN_IF_ERROR(c->WithRankAtMost(mod, 2, &mod));
        c->set_output(0, val);
        return ::tensorflow::Status::OK();
    });
