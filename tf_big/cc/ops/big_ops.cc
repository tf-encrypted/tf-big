#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("BigImport")
    .Attr("dtype: {int32, string, uint8}")
    .Input("in: dtype")
    .Output("val: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 2, &output));
      c->set_output(0, output);
      return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigImportLimbs")
   .Attr("dtype: {uint8, int32}")
   .Input("in: dtype")
   .Output("val: variant")
   .SetIsStateful()
   .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
     ::tensorflow::shape_inference::ShapeHandle inputShape;
     TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &inputShape));
     ::tensorflow::shape_inference::ShapeHandle out = c->MakeShape({c->Dim(inputShape, 1), c->Dim(inputShape, 2)});
     c->set_output(0, out);
     return ::tensorflow::Status::OK();
   });

REGISTER_OP("BigExport")
    .Attr("dtype: {int32, string, uint8}")
    .Input("val: variant")
    .Output("out: dtype")
    .SetIsStateful()
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("BigExportLimbs")
   .Attr("dtype: {int32, uint8}")
   .Input("maxval: int32")
   .Input("in: variant")
   .Output("out: dtype")
   .SetIsStateful()
   .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
     ::tensorflow::shape_inference::ShapeHandle maxvalShape;
     TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &maxvalShape));

     ::tensorflow::shape_inference::ShapeHandle inputShape;
     TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &inputShape));

     ::tensorflow::shape_inference::ShapeHandle out = c->MakeShape({c->Dim(inputShape, 1), c->Dim(inputShape, 2), c->UnknownDim()});
     c->set_output(0, out);
     return ::tensorflow::Status::OK();
   });
 

REGISTER_OP("BigRandomUniform")
    .Input("shape: int32")
    .Input("maxval: variant")
    .Output("out: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // TODO(Morten) `maxval` should be a scalar
      ::tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigRandomRsaModulus")
    .Input("bitlength: int32")
    .Output("p: variant")
    .Output("q: variant")
    .Output("n: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle bitlength_shape = c->input(0);
      ::tensorflow::shape_inference::ShapeHandle scalar_shape =
          c->MakeShape({});
      TF_RETURN_IF_ERROR(c->WithRank(bitlength_shape, 0, &bitlength_shape));
      c->set_output(0, scalar_shape);
      c->set_output(1, scalar_shape);
      c->set_output(2, scalar_shape);
      return ::tensorflow::Status::OK();
    });

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
      // NOTE: Bug - without this condition returns shape of [1,1,1,1]
      if ((c->Rank(val0) == 0) & (c->Rank(val1) == 0)) {
        c->set_output(0, c->MakeShape({1, 1}));
      } else {
        TF_RETURN_IF_ERROR(c->Merge(val0, val1, &res));
        c->set_output(0, res);
      }
      return ::tensorflow::Status::OK();
    });

REGISTER_OP("BigDiv")
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

REGISTER_OP("BigInv")
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
