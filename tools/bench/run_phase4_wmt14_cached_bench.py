"""Run the existing source-bound benchmark with the explicit WMT14 cache binary."""
import run_phase4_wmt14_bench as bench

if __name__ == '__main__':
    bench.BENCH = bench.ROOT / 'target/release/examples/wmt14_cached_perf_screen'
    bench.main()
