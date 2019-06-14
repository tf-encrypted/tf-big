cc_library(
   name = "lib",
   srcs = select({
           "@bazel_tools//src/conditions:darwin": [
                "lib/libgmp.dylib",
                "lib/libgmpxx.dylib",
           ],
           "@bazel_tools//src/conditions:linux_x86_64": [
                "lib/libgmp.so",
                "lib/libgmpxx.so",
           ]
       }) + ["include/gmp.h"],
   visibility = ["//visibility:public"],
)