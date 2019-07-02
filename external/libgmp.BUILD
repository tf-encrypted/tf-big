cc_library(
   name = "lib",
   srcs = select({
           "@bazel_tools//src/conditions:darwin": [
                "lib/libgmp.a",
                "lib/libgmpxx.a",
           ],
           "//conditions:default": [
                "lib/libgmpxx.a",
                "lib/libgmp.a",
           ]
       }),
   hdrs = ["include/gmp.h", "include/gmpxx.h"],
   visibility = ["//visibility:public"],
   strip_include_prefix = "include"
)
