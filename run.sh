#!/usr/bin/env bash
# AlphaGPT-Routine 本地一键运行
# 用法:
#   ./run.sh                 # 使用 config.env 中的配置
#   ./run.sh 600519          # 临时覆盖标的代码
#   TRAIN_ITERATIONS=1 ./run.sh   # 临时覆盖其它参数
set -euo pipefail
cd "$(dirname "$0")"

# 1. 加载本地配置（存在才加载，且不覆盖命令行已传入的同名环境变量）
if [[ -f config.env ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < config.env
fi

# 2. 可选：第一个参数覆盖 INDEX_CODE
if [[ $# -ge 1 && -n "${1:-}" ]]; then export INDEX_CODE="$1"; fi

# 3. 检查虚拟环境
PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "未找到虚拟环境，请先执行："
  echo "  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

# 4. matplotlib 缓存目录（避免写入 ~/.matplotlib 受限）
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_config}"
mkdir -p "$MPLCONFIGDIR"

echo "===> INDEX_CODE=$INDEX_CODE  START_DATE=$START_DATE  END_DATE=$END_DATE"
echo "===> TRAIN_ITERATIONS=$TRAIN_ITERATIONS  BATCH_SIZE=$BATCH_SIZE  FORCE_TRAIN=$FORCE_TRAIN"
exec "$PY" times_astock.py
