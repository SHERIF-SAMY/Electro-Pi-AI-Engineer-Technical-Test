import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from ollama import Client

from metrics import MemoryMonitor, calculate_throughput, calculate_latency_ms
from quality_eval import evaluate_rouge

def load_prompts(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_benchmark(model_name: str, prompts: list, output_dir: str):
    print(f"\n--- Starting benchmark for model: {model_name} ---")
    
    # Initialize explicit client to avoid localhost IPv6 resolution issues on Windows
    client = Client(host='http://127.0.0.1:11434')
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "peak_ram_mb": 0.0,
        "prompts": []
    }
    
    # We monitor memory across the whole model run to capture peak RAM.
    with MemoryMonitor() as monitor:
        for p in tqdm(prompts, desc=f"Evaluating {model_name}"):
            prompt_text = p['prompt']
            reference = p['reference_answer']
            
            # Run inference
            try:
                response = client.generate(model=model_name, prompt=prompt_text)
            except Exception as e:
                print(f"\nError running model {model_name}: {e}")
                continue
                
            generated_text = response.get('response', '')
            eval_duration_ns = response.get('eval_duration', 0)
            eval_count = response.get('eval_count', 0)
            
            latency_ms = calculate_latency_ms(eval_duration_ns)
            throughput = calculate_throughput(eval_count, eval_duration_ns)
            rouge_l = evaluate_rouge(generated_text, reference)
            
            results["prompts"].append({
                "id": p['id'],
                "category": p['category'],
                "generated_text": generated_text,
                "latency_ms": latency_ms,
                "throughput_tok_sec": throughput,
                "rouge_l_f1": rouge_l,
                "quality_flag_passed": True # Default to true, human review needed
            })
            
        # Get the peak memory tracked during the inference loop
        results["peak_ram_mb"] = monitor.peak_memory_mb
        
    # Save results
    safe_model_name = model_name.replace(':', '_')
    out_file = os.path.join(output_dir, f"{safe_model_name}_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {out_file}")
    if results["peak_ram_mb"] > 0:
        print(f"Peak RAM during inference: {results['peak_ram_mb']:.2f} MB\n")
    else:
        print("Peak RAM: N/A (could not track ollama server process)\n")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Run LLM Quantization Benchmark")
    parser.add_argument("--baseline", type=str, default="qwen2.5:0.5b", help="Baseline model name in Ollama")
    parser.add_argument("--quantized", type=str, default="qwen2.5:0.5b-instruct-q2_k", help="Quantized model name in Ollama")
    parser.add_argument("--prompts", type=str, default="../prompts/prompts.json", help="Path to prompts JSON file relative to script")
    parser.add_argument("--outdir", type=str, default="../benchmarks/raw_results", help="Output directory for raw results relative to script")
    args = parser.parse_args()

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, args.prompts)
    out_dir = os.path.join(script_dir, args.outdir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found at {prompts_path}")
        return
        
    prompts = load_prompts(prompts_path)
    
    # Run Baseline
    run_benchmark(args.baseline, prompts, out_dir)
    
    # Run Quantized
    run_benchmark(args.quantized, prompts, out_dir)

if __name__ == "__main__":
    main()
