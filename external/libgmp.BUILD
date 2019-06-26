cc_library(
   name = "lib",
   srcs = select({
           "@bazel_tools//src/conditions:darwin": [
                "lib/libgmp.dylib",
                "lib/libgmpxx.dylib"
           ],
           "//conditions:default": [
                "lib/libgmp.so",
                "lib/libgmpxx.so"
           ]
       }),
   hdrs = ["include/gmp.h", "include/gmpxx.h"],
   visibility = ["//visibility:public"],
   strip_include_prefix = "include"
)
