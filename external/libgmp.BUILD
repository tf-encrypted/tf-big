cc_library(
   name = "lib",
   srcs = select({
           "@bazel_tools//src/conditions:darwin": [
                "lib/libgmp.dylib",
                "lib/libgmpxx.dylib",
           ],
           "//conditions:default": [
                "lib/libgmp.so",
                "lib/libgmpxx.so",
           ]
       }) + ["include/gmp.h"],
   visibility = ["//visibility:public"],
)