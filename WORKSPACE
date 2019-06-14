load("//tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

new_local_repository(
    name = "libgmp",
    path = "/usr/local/",
    build_file = "external/libgmp.BUILD"
)
