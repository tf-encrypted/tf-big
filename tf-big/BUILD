licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_tf_big_ops.so',
    srcs = [
        "cc/kernels/tf_big_kernels.cc",
        "cc/ops/tf_big_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@libgmp//:lib"
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    linkopts = ["-lgmp", "-lgmpxx"]
)

py_library(
    name = "tf_big_ops_py",
    srcs = ([
        "python/ops/tf_big_ops.py",
    ]),
    data = [
        ":python/ops/_tf_big_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tf_big_ops_py_test",
    srcs = [
        "python/ops/tf_big_ops_test.py"
    ],
    main = "python/ops/tf_big_ops_test.py",
    deps = [
        ":tf_big_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "tf_big_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":tf_big_ops_py"
    ],
    srcs_version = "PY2AND3",
)
