#include <os/log.h>
#include <os/signpost.h>
#include <stdint.h>

/* Diagnostic-only shim. The benchmark initializes once and emits nonoverlapping
 * intervals from its main thread. No model arrays or token values are logged. */
static os_log_t marker_log;

void ncm_marker_init(void) {
    marker_log = os_log_create("org.ncm.phase2", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

void ncm_marker_begin(uint32_t batch, uint32_t sequence) {
    os_signpost_interval_begin(marker_log, OS_SIGNPOST_ID_EXCLUSIVE, "ncm_block",
                               "batch=%{public}u sequence=%{public}u", batch, sequence);
}

void ncm_marker_graph_end(uint32_t batch, uint32_t sequence) {
    os_signpost_event_emit(marker_log, OS_SIGNPOST_ID_EXCLUSIVE, "ncm_graph_end",
                           "batch=%{public}u sequence=%{public}u", batch, sequence);
}

void ncm_marker_end(uint32_t batch, uint32_t sequence) {
    os_signpost_interval_end(marker_log, OS_SIGNPOST_ID_EXCLUSIVE, "ncm_block",
                             "batch=%{public}u sequence=%{public}u", batch, sequence);
}
