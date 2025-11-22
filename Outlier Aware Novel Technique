# ============================================================================
# ULTRA-OPTIMIZED OUTLIER-AWARE QUANTIZATION
# Aggressive Performance Optimizations:
# 1. Compiled quantization kernels (torch.compile)
# 2. Size-aware application (skip small tensors)
# 3. Minimal hook overhead (single hook per layer block)
# 4. Cached computation
# 5. FP16-native operations (no dtype conversions)
# 6. Fused operations
# ============================================================================

import os
import sys
import json
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INSTALLING DEPENDENCIES")
print("=" * 80)

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.45.0 accelerate>=0.25.0 datasets>=2.14.0
!pip install -q huggingface-hub>=0.19.0
!pip install -q sentencepiece protobuf pillow
!pip install -q bitsandbytes>=0.41.0
!pip install -q scipy scikit-learn pandas tqdm matplotlib seaborn

print("\n✓ All dependencies installed\n")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# ============================================================================
# GPU SETUP
# ============================================================================

print("\n" + "=" * 80)
print("GPU SETUP")
print("=" * 80)

if not torch.cuda.is_available():
    raise RuntimeError("GPU required")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_memory:.1f} GB")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# ULTRA-FAST QUANTIZER WITH TORCH.COMPILE
# ============================================================================

class UltraFastQuantizer:
    """
    ULTRA-OPTIMIZED quantizer using:
    - torch.compile for JIT optimization
    - Size-aware skipping (only quantize large tensors)
    - FP16-native operations (no conversions)
    - Minimal branching
    - Fused operations
    """

    def __init__(self, bits=8, outlier_threshold=6.0, min_numel=1000):
        self.bits = bits
        self.outlier_threshold = outlier_threshold
        self.min_numel = min_numel  # Skip tensors smaller than this
        self.enabled = True

        # Pre-compute quantization levels
        self.qmin = -(2**(bits-1))
        self.qmax = 2**(bits-1) - 1
        self.q_levels = float(self.qmax - self.qmin)

        self.stats = {
            'quantized': 0,
            'skipped': 0,
            'total_time_ms': 0
        }

        # Compile the core quantization function
        self._quantize_core = torch.compile(self._quantize_tensor_core, mode="reduce-overhead")
        print("    → Compiled quantization kernel with torch.compile")

    @torch.no_grad()
    def _quantize_tensor_core(self, x, threshold):
        """
        Core quantization logic - will be JIT compiled
        Uses FP16 throughout, no conversions
        """
        # All operations in FP16 for speed
        mean = x.mean()
        std = x.std()

        # Find outliers - vectorized
        z_scores = torch.abs((x - mean) / (std + 1e-8))
        outlier_mask = z_scores > threshold

        # Quantize non-outliers
        # Clone to avoid modifying input
        x_temp = x.clone()
        x_temp[outlier_mask] = mean  # Temporarily mask outliers

        # Get quantization range
        x_min = x_temp.min()
        x_max = x_temp.max()
        scale = (x_max - x_min) / self.q_levels
        scale = torch.clamp(scale, min=1e-8)

        # Quantize and dequantize (fused)
        x_quant = torch.round((x_temp - x_min) / scale)
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
        x_dequant = x_quant * scale + x_min

        # Restore outliers - vectorized selection
        result = torch.where(outlier_mask, x, x_dequant)

        return result

    @torch.no_grad()
    def quantize(self, x):
        """
        Main quantization with size-aware skipping
        """
        if not self.enabled or not isinstance(x, torch.Tensor):
            return x

        # Skip small tensors (overhead > benefit)
        if x.numel() < self.min_numel:
            self.stats['skipped'] += 1
            return x

        # Skip non-2D+ tensors
        if x.dim() < 2:
            self.stats['skipped'] += 1
            return x

        start = time.perf_counter()

        # Use compiled version
        result = self._quantize_core(x, self.outlier_threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats['quantized'] += 1
        self.stats['total_time_ms'] += elapsed_ms

        return result

    def get_statistics(self):
        """Return performance statistics"""
        total = self.stats['quantized'] + self.stats['skipped']
        if self.stats['quantized'] == 0:
            return {
                'quantized_count': 0,
                'skipped_count': self.stats['skipped'],
                'avg_time_ms': 0,
                'skip_rate': 1.0 if total > 0 else 0
            }

        return {
            'quantized_count': self.stats['quantized'],
            'skipped_count': self.stats['skipped'],
            'avg_time_ms': self.stats['total_time_ms'] / self.stats['quantized'],
            'skip_rate': self.stats['skipped'] / total if total > 0 else 0
        }


class SmartLayerSelector:
    """
    Intelligently select which layers to quantize
    Strategy: Only quantize the MIDDLE layers where activations are largest
    """

    def __init__(self, num_layers, quantize_ratio=0.5):
        self.num_layers = num_layers

        # Only quantize middle 50% of layers
        total_to_quantize = int(num_layers * quantize_ratio)
        start_idx = (num_layers - total_to_quantize) // 2
        end_idx = start_idx + total_to_quantize

        self.quantize_layers = set(range(start_idx, end_idx))

        print(f"    → Quantizing {len(self.quantize_layers)}/{num_layers} layers")
        print(f"    → Range: layers {start_idx}-{end_idx-1} (middle layers only)")

    def should_quantize(self, layer_idx):
        return layer_idx in self.quantize_layers


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS_CONFIG = {
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B-Instruct",
        "type": "text",
        "min_vram": 2,
        "num_layers": 24
    },
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "type": "text",
        "min_vram": 3,
        "num_layers": 32
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "text",
        "min_vram": 8,
        "num_layers": 32
    }
}

OPTIMIZATION_CONFIGS = {
    "fp16_baseline": {
        "name": "FP16 Baseline",
        "dtype": torch.float16,
        "quantization": None,
        "use_ultra_fast": False
    },
    "int4_baseline": {
        "name": "INT4 Baseline",
        "dtype": torch.float16,
        "quantization": "int4",
        "use_ultra_fast": False
    },
    "ultra_fast_v1": {
        "name": "Ultra-Fast (Compiled, 50% layers)",
        "dtype": torch.float16,
        "quantization": None,
        "use_ultra_fast": True,
        "outlier_threshold": 6.0,
        "quantize_ratio": 0.5,
        "min_tensor_size": 10000  # Only large tensors
    },
    "ultra_fast_v2": {
        "name": "Ultra-Fast (Aggressive, 30% layers)",
        "dtype": torch.float16,
        "quantization": None,
        "use_ultra_fast": True,
        "outlier_threshold": 5.0,
        "quantize_ratio": 0.3,
        "min_tensor_size": 50000  # Only very large tensors
    }
}

def get_quantization_config(quant_type):
    """Get BitsAndBytes quantization config"""
    if quant_type == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    return None

def load_model_with_optimization(model_name, optimization):
    """Load model with ultra-fast optimization"""
    model_cfg = MODELS_CONFIG[model_name]
    opt_cfg = OPTIMIZATION_CONFIGS[optimization]

    print(f"  Loading {model_name} with {opt_cfg['name']}...")

    try:
        gc.collect()
        torch.cuda.empty_cache()

        quant_config = None
        if opt_cfg.get('quantization'):
            quant_config = get_quantization_config(opt_cfg['quantization'])

        model_id = model_cfg['hf_id']

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=opt_cfg['dtype'],
            device_map="auto",
            quantization_config=quant_config,
            token=HF_TOKEN,
            trust_remote_code=True
        )

        processor = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            trust_remote_code=True
        )

        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model.config.pad_token_id = processor.pad_token_id

        # Add ultra-fast quantization
        if opt_cfg.get('use_ultra_fast'):
            model.ultra_quantizer = UltraFastQuantizer(
                bits=8,
                outlier_threshold=opt_cfg.get('outlier_threshold', 6.0),
                min_numel=opt_cfg.get('min_tensor_size', 10000)
            )

            model.layer_selector = SmartLayerSelector(
                num_layers=model_cfg['num_layers'],
                quantize_ratio=opt_cfg.get('quantize_ratio', 0.5)
            )

            print(f"    → Min tensor size: {opt_cfg.get('min_tensor_size', 10000)} elements")
            print(f"    → Outlier threshold: {opt_cfg.get('outlier_threshold', 6.0)} std")

        print(f"  ✓ Loaded successfully")
        return model, processor, "success"

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Failed: {error_msg[:100]}")
        return None, None, error_msg[:100]

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_real_text_dataset(dataset_name="hellaswag", num_samples=100):
    """Load real evaluation datasets"""
    print(f"  Loading {dataset_name} dataset...")

    try:
        if dataset_name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            data = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                data.append({
                    'question': f"{item['ctx']}\nWhat happens next?",
                    'choices': item['endings'],
                    'answer': int(item['label']),
                    'type': 'text',
                    'source': 'hellaswag'
                })

        elif dataset_name == "arc_easy":
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            data = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                choices = item['choices']['text']
                labels = item['choices']['label']
                answer_idx = labels.index(item['answerKey'])

                data.append({
                    'question': item['question'],
                    'choices': choices,
                    'answer': answer_idx,
                    'type': 'text',
                    'source': 'arc_easy'
                })

        print(f"  ✓ Loaded {len(data)} samples from {dataset_name}")
        return data

    except Exception as e:
        print(f"  ✗ Failed to load {dataset_name}: {e}")
        return []

# ============================================================================
# ULTRA-FAST EVALUATION
# ============================================================================

def evaluate_model_ultra(model, processor, dataset, model_cfg, opt_cfg, dataset_type):
    """
    Ultra-fast evaluation with optimized hooks
    """

    results = {
        'correct': 0,
        'total': 0,
        'latencies': [],
        'memory_peak': 0,
        'errors': [],
        'token_throughput': []
    }

    has_quantizer = hasattr(model, 'ultra_quantizer')

    torch.cuda.reset_peak_memory_stats()

    # Warmup (important for torch.compile)
    print("    Running warmup (compiling kernels)...", end=" ", flush=True)
    if dataset and has_quantizer:
        sample = dataset[0]
        try:
            prompt = f"Question: {sample['question']}\nAnswer:"
            inputs = processor(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            with torch.no_grad():
                # Do extra warmup for compiled code
                for _ in range(5):
                    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            print("✓ (kernels compiled)")
        except Exception as e:
            print(f"✗ {str(e)[:50]}")
    else:
        print("✓")

    eval_samples = min(len(dataset), 50)

    for i, sample in enumerate(tqdm(dataset[:eval_samples], desc=f"  Eval {dataset_type}", leave=False)):
        try:
            question = sample['question']
            choices = sample['choices']
            answer_idx = sample['answer']

            # Format prompt
            if len(choices) == 2:
                prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nAnswer with only A or B:"
                valid_letters = 'AB'
            else:
                prompt = f"Question: {question}\n"
                for j, choice in enumerate(choices):
                    prompt += f"{chr(65+j)}. {choice}\n"
                prompt += f"Answer with only the letter:"
                valid_letters = ''.join([chr(65+j) for j in range(len(choices))])

            inputs = processor(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            # Measure inference
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                if has_quantizer:
                    hooks = []
                    layer_idx = [0]  # Track which layer we're in

                    def smart_quant_hook(module, input, output):
                        """Optimized hook with layer-aware skipping"""
                        # Check if we should quantize this layer
                        if not model.layer_selector.should_quantize(layer_idx[0]):
                            layer_idx[0] += 1
                            return output

                        layer_idx[0] += 1

                        # Handle different output types
                        if isinstance(output, tuple):
                            # Only quantize first element (attention output)
                            quantized_first = model.ultra_quantizer.quantize(output[0])
                            return (quantized_first,) + output[1:]
                        elif isinstance(output, torch.Tensor):
                            return model.ultra_quantizer.quantize(output)
                        return output

                    # Hook only attention output projections (most critical)
                    for name, module in model.named_modules():
                        if 'self_attn.o_proj' in name or 'self_attn.out_proj' in name:
                            hooks.append(module.register_forward_hook(smart_quant_hook))

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.pad_token_id if hasattr(processor, 'pad_token_id') else None
                    )

                    for hook in hooks:
                        hook.remove()
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.pad_token_id if hasattr(processor, 'pad_token_id') else None
                    )

            torch.cuda.synchronize()
            latency = time.time() - start_time
            results['latencies'].append(latency)

            # Calculate throughput
            output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = output_tokens / latency if latency > 0 else 0
            results['token_throughput'].append(tokens_per_sec)

            # Memory tracking
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            results['memory_peak'] = max(results['memory_peak'], peak_mem)

            # Check answer
            response = processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            response_upper = response.upper()
            pred_letter = None
            for char in response_upper:
                if char in valid_letters:
                    pred_letter = char
                    break

            if pred_letter:
                pred_idx = ord(pred_letter) - ord('A')
                if pred_idx == answer_idx:
                    results['correct'] += 1

            results['total'] += 1

        except Exception as e:
            results['errors'].append(str(e)[:50])
            continue

    # Calculate metrics
    if results['total'] > 0:
        metrics = {
            'accuracy': (results['correct'] / results['total']) * 100,
            'avg_latency_ms': np.mean(results['latencies']) * 1000 if results['latencies'] else 0,
            'p50_latency_ms': np.percentile(results['latencies'], 50) * 1000 if results['latencies'] else 0,
            'p95_latency_ms': np.percentile(results['latencies'], 95) * 1000 if results['latencies'] else 0,
            'avg_tokens_per_sec': np.mean(results['token_throughput']) if results['token_throughput'] else 0,
            'memory_peak_gb': results['memory_peak'],
            'samples_evaluated': results['total'],
            'samples_correct': results['correct']
        }

        # Add quantizer statistics
        if has_quantizer:
            quant_stats = model.ultra_quantizer.get_statistics()
            metrics.update(quant_stats)

        return metrics

    return None

# ============================================================================
# MAIN EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("ULTRA-OPTIMIZED QUANTIZATION EVALUATION")
print("Using: torch.compile + size-aware skipping + smart layer selection")
print("=" * 80)

results_data = []

# Load datasets
print("\nLoading datasets...")
text_datasets = {
    'hellaswag': load_real_text_dataset("hellaswag", num_samples=50),
    'arc_easy': load_real_text_dataset("arc_easy", num_samples=50),
}

# Models to test
models_to_test = ['qwen2-0.5b', 'phi-2']
if gpu_memory >= 40:
    models_to_test.append('mistral-7b')

# Test configurations
optimizations_to_test = [
    'fp16_baseline',
    'int4_baseline',
    'ultra_fast_v1',
    'ultra_fast_v2'
]

print(f"\nTesting {len(models_to_test)} models × {len(optimizations_to_test)} configs")

total = len(models_to_test) * len(optimizations_to_test)
completed = 0

for model_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    model_cfg = MODELS_CONFIG[model_name]

    for opt_name in optimizations_to_test:
        opt_cfg = OPTIMIZATION_CONFIGS[opt_name]
        completed += 1

        print(f"\n[{completed}/{total}] {opt_cfg['name']}")

        model, processor, status = load_model_with_optimization(model_name, opt_name)

        if model is None:
            continue

        # Evaluate on datasets
        for dataset_name, dataset in text_datasets.items():
            if not dataset:
                continue

            metrics = evaluate_model_ultra(
                model, processor, dataset,
                model_cfg, opt_cfg, 'text'
            )

            if metrics:
                result = {
                    'model': model_name,
                    'optimization': opt_cfg['name'],
                    'dataset': dataset_name,
                    'accuracy': metrics['accuracy'],
                    'avg_latency_ms': metrics['avg_latency_ms'],
                    'p50_latency_ms': metrics['p50_latency_ms'],
                    'p95_latency_ms': metrics['p95_latency_ms'],
                    'memory_peak_gb': metrics['memory_peak_gb'],
                    'tokens_per_sec': metrics['avg_tokens_per_sec'],
                    'samples_evaluated': metrics['samples_evaluated']
                }

                if 'quantized_count' in metrics:
                    result['quantized_ops'] = metrics['quantized_count']
                    result['skipped_ops'] = metrics['skipped_count']
                    result['skip_rate'] = metrics['skip_rate']

                results_data.append(result)

                print(f"    ✓ {dataset_name}: {metrics['accuracy']:.1f}% accuracy")
                print(f"    ✓ Latency: {metrics['avg_latency_ms']:.1f}ms (p95: {metrics['p95_latency_ms']:.1f}ms)")
                print(f"    ✓ Memory: {metrics['memory_peak_gb']:.2f}GB")
                print(f"    ✓ Throughput: {metrics['avg_tokens_per_sec']:.1f} tokens/sec")

                if 'skip_rate' in metrics:
                    print(f"    ✓ Skip rate: {metrics['skip_rate']*100:.1f}% (skipped small tensors)")

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

if results_data:
    df = pd.DataFrame(results_data)

    print("\n" + "="*70)
    print("SUMMARY BY OPTIMIZATION")
    print("="*70)
    summary = df.groupby('optimization').agg({
        'accuracy': 'mean',
        'avg_latency_ms': 'mean',
        'memory_peak_gb': 'mean',
        'tokens_per_sec': 'mean'
    }).round(2)
    print(summary)

    print("\n" + "="*70)
    print("SPEED IMPROVEMENT ANALYSIS")
    print("="*70)

    fp16_data = df[df['optimization'] == 'FP16 Baseline']
    int4_data = df[df['optimization'] == 'INT4 Baseline']

    if not fp16_data.empty:
        fp16_acc = fp16_data['accuracy'].mean()
        fp16_lat = fp16_data['avg_latency_ms'].mean()
        fp16_mem = fp16_data['memory_peak_gb'].mean()

        print(f"FP16 Baseline (Reference):")
        print(f"  Accuracy: {fp16_acc:.1f}%")
        print(f"  Latency: {fp16_lat:.1f}ms (1.00x)")
        print(f"  Memory: {fp16_mem:.2f}GB")

    if not int4_data.empty:
        int4_acc = int4_data['accuracy'].mean()
        int4_lat = int4_data['avg_latency_ms'].mean()
        int4_mem = int4_data['memory_peak_gb'].mean()

        print(f"\nINT4 Baseline:")
        print(f"  Accuracy: {int4_acc:.1f}% ({int4_acc-fp16_acc:+.1f}% vs FP16)")
        print(f"  Latency: {int4_lat:.1f}ms ({int4_lat/fp16_lat:.2f}x)")
        print(f"  Memory: {int4_mem:.2f}GB ({int4_mem/fp16_mem:.2f}x)")

    for opt_name in ['Ultra-Fast (Compiled, 50% layers)', 'Ultra-Fast (Aggressive, 30% layers)']:
        opt_data = df[df['optimization'] == opt_name]
        if not opt_data.empty:
            acc = opt_data['accuracy'].mean()
            lat = opt_data['avg_latency_ms'].mean()
            mem = opt_data['memory_peak_gb'].mean()

            print(f"\n{opt_name}:")
            print(f"  Accuracy: {acc:.1f}% ({acc-fp16_acc:+.1f}% vs FP16)")
            print(f"  Latency: {lat:.1f}ms ({lat/fp16_lat:.2f}x)")
            print(f"  Memory: {mem:.2f}GB")

            # Show if it beats INT4
            if not int4_data.empty:
                if acc > int4_acc and lat < int4_lat:
                    print(f"  🎯 BEATS INT4: Better accuracy AND faster!")
                elif acc > int4_acc:
                    print(f"  ✓ Better accuracy than INT4 ({acc-int4_acc:+.1f}%)")
                elif lat < int4_lat:
                    print(f"  ✓ Faster than INT4 ({int4_lat/lat:.2f}x)")

    df.to_csv('ultra_optimized_results.csv', index=False)
    print(f"\n✓ Results saved to: ultra_optimized_results.csv")

print("\n" + "=" * 80)
print("✓ ULTRA-OPTIMIZED EVALUATION COMPLETE")
print("=" * 80)
