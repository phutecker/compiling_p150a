#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
LOG_DIR="${RESULTS_DIR}/logs"
CSV_PATH="${RESULTS_DIR}/results.csv"
TT_SNAPSHOT="/home/ttpatrick/.local/share/containers/storage/overlay/9506a74f929155ed2bd932965de53bcc35a496e042b2fc590abfb5ae126947e3/diff/root/.cache/uv/archive-v0/ic5dQ7xXBxV5fehfYv7kr/ttnn"
METALIUM_IMAGE="ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc"
RUNTIME_DIAG_LOG="${LOG_DIR}/runtime_diagnostics.log"

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
        log "Checking /proc/driver/tenstorrent/0/pids..."
        if [[ -r /proc/driver/tenstorrent/0/pids ]]; then
            cat /proc/driver/tenstorrent/0/pids
        else
            log "WARNING: could not read /proc/driver/tenstorrent/0/pids"
        fi

        log "Checking 1G hugepage availability..."
        for metric in free_hugepages nr_hugepages nr_hugepages_mempolicy resv_hugepages surplus_hugepages; do
            local path="/sys/kernel/mm/hugepages/hugepages-1048576kB/${metric}"
            if [[ -r "${path}" ]]; then
                printf '%s:%s\n' "${metric}" "$(cat "${path}")"
            else
                printf '%s:%s\n' "${metric}" "unreadable"
            fi
        done

        log "Checking running tt-metalium containers..."
        if ! docker ps --no-trunc \
            --filter "ancestor=${METALIUM_IMAGE}" \
            --format '{{.ID}}|{{.Image}}|{{.Status}}|{{.Names}}|{{.Command}}'; then
            log "WARNING: docker ps failed"
        fi
    } > "${RUNTIME_DIAG_LOG}" 2>&1
}

has_supported_host_surface() {
    local api_dir="${TT_SNAPSHOT}/tt_metal/api/tt-metalium"
    local required_headers=(
        "${TT_SNAPSHOT}/tt_metal/api/tt-metalium/host_api.hpp"
        "${TT_SNAPSHOT}/tt_metal/api/tt-metalium/device.hpp"
        "${TT_SNAPSHOT}/tt_metal/api/tt-metalium/program.hpp"
        "${TT_SNAPSHOT}/tt_metal/api/tt-metalium/tt_metal.hpp"
    )

    for header in "${required_headers[@]}"; do
        if [[ -f "${header}" ]]; then
            return 0
        fi
    done

    if [[ ! -d "${api_dir}" ]]; then
        return 1
    fi

    return 1
}

log "Locating tt-metal example..."
if [[ ! -d "${TT_SNAPSHOT}" ]]; then
    log "ERROR: local tt-metal snapshot was not found at ${TT_SNAPSHOT}"
    exit 1
fi
log "Found local tt-metal snapshot: ${TT_SNAPSHOT}"

log "Writing fresh results CSV..."
printf 'variant,n_tiles,status,max_abs_err,max_rel_err,time_ms,notes\n' > "${CSV_PATH}"

log "Showing copied baseline sources..."
find "${ROOT_DIR}/baseline" -type f | sort

log "Showing copied batch2 sources..."
find "${ROOT_DIR}/batch2" -type f | sort

log "Running legality check..."
if ! "${ROOT_DIR}/scripts/legality_check.py" 2>&1 | tee "${LOG_DIR}/legality_check.log"; then
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
    log "ERROR: runtime failed"
    log "ERROR: tt-smi could not detect a Tenstorrent device"
    cat "${LOG_DIR}/tt_smi_list.log"
    write_failure_rows "baseline" "no_device" "tt-smi could not detect Tenstorrent hardware"
    write_failure_rows "batch2" "no_device" "tt-smi could not detect Tenstorrent hardware"
    log "Benchmarking complete..."
    log "Final CSV:"
    cat "${CSV_PATH}"
    exit 0
fi

capture_runtime_diagnostics
log "Runtime diagnostics written to ${RUNTIME_DIAG_LOG}"

RUNNING_METALIUM_CONTAINERS="$(docker ps --no-trunc --filter "ancestor=${METALIUM_IMAGE}" \
    --format '{{.ID}} {{.Status}} {{.Names}}' || true)"
if [[ -n "${RUNNING_METALIUM_CONTAINERS}" ]]; then
    log "WARNING: existing tt-metalium container(s) are still running"
    printf '%s\n' "${RUNNING_METALIUM_CONTAINERS}"
    log "WARNING: refusing to launch another tt-metalium runtime while those containers may still hold TLB mappings"
    write_failure_rows "baseline" "runtime_busy" "existing tt-metalium container may still hold device mappings; see results/logs/runtime_diagnostics.log"
    write_failure_rows "batch2" "runtime_busy" "existing tt-metalium container may still hold device mappings; see results/logs/runtime_diagnostics.log"
    log "Benchmarking complete..."
    log "Final CSV:"
    cat "${CSV_PATH}"
    exit 0
fi

log "Running tt-metalium device smoke test..."
set +e
tt-metalium -lc "python3 /home/user/compiling_p150a/scripts/ttnn_smoke.py --mode open_device" \
    > "${LOG_DIR}/ttnn_open_device.log" 2>&1
TTNN_SMOKE_STATUS=$?
set -e

if [[ ${TTNN_SMOKE_STATUS} -ne 0 ]]; then
    log "ERROR: runtime failed"
    log "ERROR: tt-metalium could not open the device cleanly"
    cat "${LOG_DIR}/ttnn_open_device.log"
    write_failure_rows "baseline" "ttnn_open_failed" "tt-metalium open_device smoke failed; inspect results/logs/ttnn_open_device.log and results/logs/runtime_diagnostics.log"
    write_failure_rows "batch2" "ttnn_open_failed" "tt-metalium open_device smoke failed; inspect results/logs/ttnn_open_device.log and results/logs/runtime_diagnostics.log"
    log "Benchmarking complete..."
    log "Final CSV:"
    cat "${CSV_PATH}"
    exit 0
fi

log "Checking for standalone host SDK headers..."
if ! has_supported_host_surface; then
    log "WARNING: build path is still blocked"
    log "WARNING: the packaged install ships kernels and libtt_metal.so but not the public standalone host headers"
    log "WARNING: no supported tiny custom-host build path was found on this machine"
    write_failure_rows "baseline" "packaged_runtime_only" "device runtime works but packaged tt-metalium does not expose a supported standalone host build surface"
    write_failure_rows "batch2" "packaged_runtime_only" "device runtime works but packaged tt-metalium does not expose a supported standalone host build surface"
    log "Benchmarking complete..."
    log "Final CSV:"
    cat "${CSV_PATH}"
    exit 0
fi

log "WARNING: a supported standalone host surface was detected unexpectedly."
log "WARNING: this repo still does not contain an audited host runner implementation for it."
write_failure_rows "baseline" "manual_follow_up_needed" "standalone host surface detected unexpectedly; audited host runner still missing"
write_failure_rows "batch2" "manual_follow_up_needed" "standalone host surface detected unexpectedly; audited host runner still missing"

log "Benchmarking complete..."
log "Final CSV:"
cat "${CSV_PATH}"
