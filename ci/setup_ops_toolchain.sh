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

# Check the host toolchain required to build paddlefleet_ops.
# DeepGEMM (via DeepJIT) needs C++20 <format> (GCC >= 13) and elfutils headers.
# The build images ship both; this is only a guard for local/other environments.
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
    if _ops_toolchain_ok gcc g++; then
        echo "[setup_ops_toolchain] Toolchain already satisfied: $(gcc -dumpfullversion)"
        return 0
    fi

    local major
    major=$(_ops_gcc_major gcc)
    if [[ -z "${major}" || "${major}" -lt ${_OPS_MIN_GCC_MAJOR} ]]; then
        echo "[setup_ops_toolchain] GCC >= ${_OPS_MIN_GCC_MAJOR} is required for C++20 <format>, found: ${major:-none}." >&2
        echo "  install g++-${_OPS_MIN_GCC_MAJOR} (apt) or use an Ubuntu 24.04 build image." >&2
    else
        echo "[setup_ops_toolchain] GCC ${major} is new enough but <elfutils/libdwfl.h> is unavailable." >&2
        echo "  install libdw-dev (apt)." >&2
    fi
    return 1
}

_setup_ops_toolchain
