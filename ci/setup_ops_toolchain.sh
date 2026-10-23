# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Prepare the host toolchain required to build paddlefleet_ops.
# DeepGEMM (via DeepJIT) needs C++20 <format> (GCC >= 13) and elfutils headers.
# Must be sourced so that PATH/CC/CXX take effect in the caller's shell:
#   source ci/setup_ops_toolchain.sh

_OPS_MIN_GCC_MAJOR=13

_ops_gcc_major() {
    "$1" -dumpversion 2>/dev/null | cut -d. -f1
}

_ops_toolchain_ok() {
    local major
    major=$(_ops_gcc_major "${1:-gcc}")
    [[ -n "${major}" && "${major}" -ge ${_OPS_MIN_GCC_MAJOR} ]] || return 1
    echo '#include <elfutils/libdwfl.h>
#include <format>' | "${2:-g++}" -std=c++20 -fsyntax-only -x c++ - >/dev/null 2>&1
}

_setup_ops_toolchain() {
    local gcc_bin="" gxx_bin=""

    if _ops_toolchain_ok gcc g++; then
        echo "[setup_ops_toolchain] Toolchain already satisfied: $(gcc -dumpfullversion)"
        return 0
    fi

    if command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
        # manylinux (EL8-based): use gcc-toolset-13 from AppStream
        local pm
        pm=$(command -v dnf || command -v yum)
        "${pm}" install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ elfutils-devel || return 1
        gcc_bin=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
        gxx_bin=/opt/rh/gcc-toolset-13/root/usr/bin/g++
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update && apt-get install -y libdw-dev || return 1
        if [[ $(_ops_gcc_major gcc) -ge ${_OPS_MIN_GCC_MAJOR} ]]; then
            gcc_bin=$(command -v gcc)
            gxx_bin=$(command -v g++)
        else
            apt-get install -y gcc-13 g++-13 || return 1
            gcc_bin=$(command -v gcc-13)
            gxx_bin=$(command -v g++-13)
        fi
    else
        echo "[setup_ops_toolchain] No supported package manager found." >&2
        return 1
    fi

    # Paddle's cpp_extension calls nvcc with `-ccbin cc` and setuptools uses
    # gcc/c++, so shadow all generic compiler names with the selected GCC.
    local shim_dir=/tmp/paddlefleet_ops_toolchain/bin
    mkdir -p "${shim_dir}"
    ln -sfn "${gcc_bin}" "${shim_dir}/cc"
    ln -sfn "${gcc_bin}" "${shim_dir}/gcc"
    ln -sfn "${gxx_bin}" "${shim_dir}/c++"
    ln -sfn "${gxx_bin}" "${shim_dir}/g++"
    export PATH="${shim_dir}:${PATH}"
    export CC="${shim_dir}/gcc"
    export CXX="${shim_dir}/g++"

    if ! _ops_toolchain_ok gcc g++; then
        echo "[setup_ops_toolchain] GCC >= ${_OPS_MIN_GCC_MAJOR} with <format> and elfutils headers is still unavailable." >&2
        return 1
    fi
    echo "[setup_ops_toolchain] Using $(command -v gcc) ($(gcc -dumpfullversion))"
}

_setup_ops_toolchain
