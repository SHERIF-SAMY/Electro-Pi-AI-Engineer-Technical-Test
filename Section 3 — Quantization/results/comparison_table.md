# Model Comparison Table

| Metric                 | Baseline (qwen2.5:0.5b)   | Quantized (qwen2.5:0.5b-instruct-q2_k)   |
|:-----------------------|:--------------------------|:-----------------------------------------|
| Precision              | Q4_K_M (4-bit)            | Q2_K (2-bit)                             |
| Model Size (disk)      | 397 MB                    | ~270 MB                                  |
| Peak RAM Usage (MB)    | 88.14                     | 124.52                                   |
| Avg Latency (ms)       | 1188.14                   | 1652.91                                  |
| Avg Throughput (tok/s) | 259.24                    | 260.77                                   |
| ROUGE-L (avg F1)       | 0.2990                    | 0.2229                                   |
| Quality Flags Passed   | 5/5                       | 5/5                                      |

> *Note: Metrics were measured locally. 'Quality Flags Passed' represents human-reviewed adherence to prompt instructions and factual correctness.*