#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT_DIR/legacy/tracker_bridge/tracker_bridge.cpp"
OUT_DIR="$ROOT_DIR/bin"
OUT="$OUT_DIR/tracker_bridge"

if [[ ! -f "$SRC" ]]; then
  echo "Missing source file: $SRC" >&2
  exit 1
fi

: "${NDI_SDK_INCLUDE_DIR:?Set NDI_SDK_INCLUDE_DIR to your NDI SDK include directory}"
: "${NDI_SDK_LIB_DIR:?Set NDI_SDK_LIB_DIR to your NDI SDK library directory}"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra -pthread}"

# Space-delimited list, for example: "CombinedApi ndicapi"
NDI_SDK_LIBS="${NDI_SDK_LIBS:-CombinedApi}"

mkdir -p "$OUT_DIR"

read -r -a SDK_LIB_ARRAY <<< "$NDI_SDK_LIBS"
LINK_LIB_ARGS=()
for lib in "${SDK_LIB_ARRAY[@]}"; do
  LINK_LIB_ARGS+=("-l${lib}")
done

set -x
"$CXX" $CXXFLAGS \
  -I"$NDI_SDK_INCLUDE_DIR" \
  "$SRC" \
  -L"$NDI_SDK_LIB_DIR" \
  "${LINK_LIB_ARGS[@]}" \
  -Wl,-rpath,"$NDI_SDK_LIB_DIR" \
  -o "$OUT"
set +x

echo "Built legacy tracker_bridge at: $OUT"
