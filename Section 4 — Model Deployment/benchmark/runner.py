import asyncio
import httpx
import time
import json
import statistics
from datetime import datetime
from pathlib import Path

# Config
API_URL = "http://localhost:8000/api/v1/stream"
PROMPT = "Explain the concept of neural networks in 3 sentences."
CONCURRENCY = 10
ROUNDS = 3
WARM_UP = 2

async def run_request(client, req_id):
    start_time = time.perf_counter()
    ttft = None
    total_tokens = 0
    error = None
    metrics = None
    
    try:
        async with client.stream("POST", API_URL, json={"prompt": PROMPT}) as response:
            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                        
                    data_str = line[len("data: "):]
                    try:
                        data = json.loads(data_str)
                        if data.get("error"):
                            error = data["error"]
                            break
                            
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                            
                        if data.get("done"):
                            metrics = data.get("metrics")
                        elif data.get("token"):
                            total_tokens += 1
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        error = str(e)
        
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return {
        "req_id": req_id,
        "ttft": ttft * 1000 if ttft else None,
        "latency": latency * 1000,
        "tokens": total_tokens,
        "error": error,
        "server_metrics": metrics
    }

async def main():
    print(f"Starting benchmark: {CONCURRENCY} concurrent requests, {ROUNDS} rounds.")
    print(f"Target: {API_URL}")
    
    timeout = httpx.Timeout(180.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print(f"Running {WARM_UP} warm-up requests...")
        await asyncio.gather(*[run_request(client, f"warmup-{i}") for i in range(WARM_UP)])
        
        all_results = []
        for r in range(ROUNDS):
            print(f"Round {r+1}/{ROUNDS}...")
            tasks = [run_request(client, f"r{r}-req{i}") for i in range(CONCURRENCY)]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)
            
    # Process results
    from benchmark.report import generate_report
    generate_report(all_results, CONCURRENCY, ROUNDS, WARM_UP)

if __name__ == "__main__":
    asyncio.run(main())
