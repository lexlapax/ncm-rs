"""Benchmark explicit cached WMT14 generation in batches of at most four."""
import run_phase4_wmt14_bench as bench

if __name__ == '__main__':
    bench.BENCH = bench.ROOT / 'target/release/examples/wmt14_cached_microbatch_perf'
    bench.main()
