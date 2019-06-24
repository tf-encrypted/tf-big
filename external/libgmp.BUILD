cc_library(
   name = "lib",
   srcs = select({
           "@bazel_tools//src/conditions:darwin": [
                "lib/libgmp.dylib"
           ],
           "//conditions:default": [
                "lib/libgmp.so"
           ]
       }) + ["include/gmp.h"],
   visibility = ["//visibility:public"],
)