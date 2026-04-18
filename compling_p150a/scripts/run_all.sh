#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
LOG_DIR="${RESULTS_DIR}/logs"
CSV_PATH="${RESULTS_DIR}/results.csv"
RUNTIME_DIAG_LOG="${LOG_DIR}/runtime_diagnostics.log"

TT_METAL_HOME="${TT_METAL_HOME:-$HOME/.local/lib/tt-metal}"
TT_METAL_BUILD="${TT_METAL_HOME}/.build/gcc"
LOCAL_CMAKE="${HOME}/.local/opt/cmake-3.31.6-linux-x86_64/bin/cmake"

COMPARE_SRC_DIR="${ROOT_DIR}/compare"
COMPARE_BUILD_DIR="${COMPARE_SRC_DIR}/build"
COMPARE_BIN="${COMPARE_BUILD_DIR}/compare_eltwise_sfpu"

mkdir -p "${LOG_DIR}"

log() {
    printf '%s\n' "$*"
}

append_csv_row() {
    local variant="$1"
    local n_tiles="$2"
    local status="$3"
    local max_abs_err="$4"
    local max_rel_err="$5"
    local time_ms="$6"
    local notes="$7"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "${variant}" "${n_tiles}" "${status}" "${max_abs_err}" "${max_rel_err}" "${time_ms}" "${notes}" \
        >> "${CSV_PATH}"
}

write_failure_rows() {
    local variant="$1"
    local status="$2"
    local notes="$3"
    for n_tiles in 1 2 32 256; do
        append_csv_row "${variant}" "${n_tiles}" "${status}" "NA" "NA" "NA" "${notes}"
    done
}

capture_runtime_diagnostics() {
    {
        log "Inspecting TT runtime diagnostics..."
        log "TT_METAL_HOME=${TT_METAL_HOME}"
        log "TT_METAL_BUILD=${TT_METAL_BUILD}"

        log "Checking /proc/driver/tenstorrent/0/pids..."
        if [[ -r /proc/driver/tenstorrent/0/pids ]]; then
            cat /proc/driver/tenstorrent/0/pids
        else
            log "WARNING: could not read /proc/driver/tenstorrent/0/pids"
        fi

        log "Checking 1G hugepage availability..."
        for metric in free_hugepages nr_hugepages nr_hugepages_mempolicy resv_hugepages surplus_hugepages; do
            path="/sys/kernel/mm/hugepages/hugepages-1048576kB/${metric}"
            if [[ -r "${path}" ]]; then
                printf '%s:%s\n' "${metric}" "$(cat "${path}")"
            else
                printf '%s:%s\n' "${metric}" "unreadable"
            fi
        done
    } > "${RUNTIME_DIAG_LOG}" 2>&1
}

run_one_variant() {
    local variant="$1"
    local n_tiles="$2"
    local repeats="$3"
    local log_path="${LOG_DIR}/${variant}_n${n_tiles}.log"

    log "Running variant=${variant} n_tiles=${n_tiles} repeats=${repeats}"

    set +e
    "${COMPARE_BIN}" \
        --repo-root "${ROOT_DIR}" \
        --variant "${variant}" \
        --n-tiles "${n_tiles}" \
        --repeats "${repeats}" \
        > "${log_path}" 2>&1
    local status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
        log "ERROR: compare runner failed for ${variant} n_tiles=${n_tiles}"
        cat "${log_path}"
        append_csv_row "${variant}" "${n_tiles}" "run_failed" "NA" "NA" "NA" "see ${log_path}"
        return 1
    fi

    local summary
    summary="$(grep "^variant=${variant} " "${log_path}" | tail -n 1 || true)"

    if [[ -z "${summary}" ]]; then
        log "ERROR: could not parse summary line for ${variant} n_tiles=${n_tiles}"
        cat "${log_path}"
        append_csv_row "${variant}" "${n_tiles}" "parse_failed" "NA" "NA" "NA" "missing summary line; see ${log_path}"
        return 1
    fi

    local avg_ms max_abs_err max_rel_err
    avg_ms="$(printf '%s\n' "${summary}" | sed -n 's/.* avg_ms=\([^ ]*\).*/\1/p')"
    max_abs_err="$(printf '%s\n' "${summary}" | sed -n 's/.* max_abs_err=\([^ ]*\).*/\1/p')"
    max_rel_err="$(printf '%s\n' "${summary}" | sed -n 's/.* max_rel_err=\([^ ]*\).*/\1/p')"

    append_csv_row "${variant}" "${n_tiles}" "ok" "${max_abs_err}" "${max_rel_err}" "${avg_ms}" "see ${log_path}"
}

log "Using TT_METAL_HOME=${TT_METAL_HOME}"
log "Using TT_METAL_BUILD=${TT_METAL_BUILD}"

if [[ ! -d "${TT_METAL_HOME}" ]]; then
    log "ERROR: TT_METAL_HOME not found: ${TT_METAL_HOME}"
    exit 1
fi

if [[ ! -f "${TT_METAL_BUILD}/tt_metal/Release/libtt_metal.so" ]]; then
    log "ERROR: libtt_metal.so not found under ${TT_METAL_BUILD}"
    exit 1
fi

if [[ ! -x "${LOCAL_CMAKE}" ]]; then
    log "ERROR: expected cmake not found at ${LOCAL_CMAKE}"
    exit 1
fi

printf 'variant,n_tiles,status,max_abs_err,max_rel_err,time_ms,notes\n' > "${CSV_PATH}"

log "Showing copied baseline sources..."
find "${ROOT_DIR}/baseline" -type f | sort

log "Showing copied batch2 sources..."
find "${ROOT_DIR}/batch2" -type f | sort

log "Running legality check..."
if ! TT_METAL_HOME="${TT_METAL_HOME}" "${ROOT_DIR}/scripts/legality_check.py" 2>&1 | tee "${LOG_DIR}/legality_check.log"; then
    log "ERROR: legality check failed"
    write_failure_rows "baseline" "legality_check_failed" "local legality check failed"
    write_failure_rows "batch2" "legality_check_failed" "local legality check failed"
    exit 1
fi

log "Checking TT device visibility..."
set +e
tt-smi -ls > "${LOG_DIR}/tt_smi_list.log" 2>&1
TT_SMI_STATUS=$?
set -e

if [[ ${TT_SMI_STATUS} -ne 0 ]]; then
    log "ERROR: tt-smi could not detect a Tenstorrent device"
    cat "${LOG_DIR}/tt_smi_list.log"
    write_failure_rows "baseline" "no_device" "tt-smi could not detect Tenstorrent hardware"
    write_failure_rows "batch2" "no_device" "tt-smi could not detect Tenstorrent hardware"
    cat "${CSV_PATH}"
    exit 0
fi

capture_runtime_diagnostics
log "Runtime diagnostics written to ${RUNTIME_DIAG_LOG}"

log "Configuring compare runner..."
rm -rf "${COMPARE_BUILD_DIR}"
TT_METAL_HOME="${TT_METAL_HOME}" \
CC=gcc-12 CXX=g++-12 \
"${LOCAL_CMAKE}" -S "${COMPARE_SRC_DIR}" -B "${COMPARE_BUILD_DIR}" \
    > "${LOG_DIR}/cmake_configure.log" 2>&1 || {
        log "ERROR: cmake configure failed"
        cat "${LOG_DIR}/cmake_configure.log"
        exit 1
    }

log "Building compare runner..."
TT_METAL_HOME="${TT_METAL_HOME}" \
CC=gcc-12 CXX=g++-12 \
"${LOCAL_CMAKE}" --build "${COMPARE_BUILD_DIR}" -j \
    > "${LOG_DIR}/cmake_build.log" 2>&1 || {
        log "ERROR: cmake build failed"
        cat "${LOG_DIR}/cmake_build.log"
        exit 1
    }

if [[ ! -x "${COMPARE_BIN}" ]]; then
    log "ERROR: compare binary not found: ${COMPARE_BIN}"
    exit 1
fi

export TT_METAL_HOME
export LD_LIBRARY_PATH="${TT_METAL_BUILD}/tt_metal/Release:${TT_METAL_BUILD}/lib:${TT_METAL_BUILD}/tt_metal/third_party/umd/device/Release:${LD_LIBRARY_PATH:-}"

for n_tiles in 2 32 256; do
    run_one_variant baseline "${n_tiles}" 5 || true
    run_one_variant batch2 "${n_tiles}" 5 || true
done

log "Benchmarking complete..."
log "Final CSV:"
cat "${CSV_PATH}"
