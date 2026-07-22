import json
import statistics
from datetime import datetime
from pathlib import Path

def generate_report(results, concurrency, rounds, warmup):
    errors = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]
    
    ttfts = [r["ttft"] for r in successes if r["ttft"]]
    latencies = [r["latency"] for r in successes]
    tokens = sum(r["tokens"] for r in successes)
    total_time = sum(r["latency"] for r in successes) / 1000 # in seconds, naive sum for throughput
    
    # Actually for true throughput we should use wall clock time of the round, 
    # but naive tokens/sec per request average is fine here.
    
    print("\n" + "═"*62)
    print("           Benchmark Report — Section 4 (Task 4.1)           ")
    print(f"           Model: qwen2.5:0.5b  |  Concurrent: {concurrency}            ")
    print("═"*62 + "\n")
    
    if not successes:
        print("All requests failed!")
        print(f"Errors: {errors}")
        return
        
    def stats(data):
        if not data: return "N/A", "N/A", "N/A", "N/A", "N/A"
        return (
            min(data), max(data), statistics.mean(data),
            statistics.quantiles(data, n=20)[18] if len(data) >= 20 else max(data), # P95 approximation
            statistics.stdev(data) if len(data) > 1 else 0
        )
        
    t_min, t_max, t_mean, t_p95, t_std = stats(ttfts)
    l_min, l_max, l_mean, l_p95, l_std = stats(latencies)
    
    print("┌────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│ Metric         │ Min (ms)   │ Max (ms)   │ Mean (ms)  │ P95 (ms)   │ StdDev     │")
    print("├────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
    print(f"│ TTFT           │ {t_min:10.1f} │ {t_max:10.1f} │ {t_mean:10.1f} │ {t_p95:10.1f} │ {t_std:10.1f} │")
    print(f"│ Total Latency  │ {l_min:10.1f} │ {l_max:10.1f} │ {l_mean:10.1f} │ {l_p95:10.1f} │ {l_std:10.1f} │")
    print("└────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n")
    
    print(f"Tokens Generated (avg per request): {tokens / len(successes):.1f} tokens")
    print(f"Error Count:                        {len(errors)} / {len(results)}")
    print(f"Success Rate:                       {(len(successes)/len(results))*100:.1f}%\n")
    
    # Save results
    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"report_{timestamp}.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "config": {"concurrency": concurrency, "rounds": rounds, "warmup": warmup},
            "summary": {
                "ttft_mean": t_mean, "latency_mean": l_mean,
                "success_rate": (len(successes)/len(results))*100
            },
            "results": results
        }, f, indent=2)
        
    print(f"Results saved to: {out_file}")
