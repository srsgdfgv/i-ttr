#!/usr/bin/env bash
# Soft-link ALOHA demo HDF5 trees into ./data without copying (~tens of GB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-}"

if [[ -z "${DATA_ROOT}" ]]; then
  cat <<EOF
Usage: $0 /path/to/aloha/data

Expected layout under DATA_ROOT:
  sim_insertion_human/
  sim_transfer_cube_human/
  (optional) sim_insertion_human_kin/
  (optional) outputs/   # ckpts; usually left on the machine, not linked

Example (this workstation):
  $0 /home/lee/projects/demospeedup/aloha/data

Example (lab server after download):
  $0 /data/share/aloha_demos
EOF
  exit 1
fi

DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
mkdir -p "${ROOT}/data"

link_one() {
  local name="$1"
  local src="${DATA_ROOT}/${name}"
  local dst="${ROOT}/data/${name}"
  if [[ ! -d "${src}" ]]; then
    echo "[skip] missing ${src}"
    return
  fi
  if [[ -L "${dst}" || -d "${dst}" ]]; then
    echo "[exists] ${dst}"
    return
  fi
  ln -s "${src}" "${dst}"
  echo "[link] ${dst} -> ${src}"
}

link_one sim_insertion_human
link_one sim_transfer_cube_human
link_one sim_insertion_scripted
link_one sim_transfer_cube_scripted
link_one sim_insertion_human_kin

# Remember path for other scripts
echo "TSR_DATA_ROOT=${DATA_ROOT}" > "${ROOT}/data/.local_paths.env"
echo "Wrote ${ROOT}/data/.local_paths.env"
ls -la "${ROOT}/data"
