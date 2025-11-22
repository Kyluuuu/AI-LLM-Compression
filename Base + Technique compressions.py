# ============================================================================
# COMPREHENSIVE MULTIMODAL MODEL EVALUATION PIPELINE
# Thesis Implementation: Quantization + Activation Quantization + TRIM
# ============================================================================

import os
import sys
import json
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DEPENDENCY INSTALLATION
# ============================================================================

print("=" * 80)
print("INSTALLING DEPENDENCIES")
print("=" * 80)

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.45.0 accelerate>=0.25.0 datasets>=2.14.0
!pip install -q huggingface-hub>=0.19.0
!pip install -q sentencepiece protobuf pillow
!pip install -q bitsandbytes>=0.41.0
!pip install -q scipy scikit-learn pandas tqdm
!pip install -q einops

print("\n✓ All dependencies installed\n")

# ============================================================================
# SECTION 2: IMPORTS AND SETUP
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed
)
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from collections import defaultdict

from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# ============================================================================
# SECTION 3: GPU AND ENVIRONMENT SETUP
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
# SECTION 4: TRIM IMPLEMENTATION
# ============================================================================

class TRIM:
    """Token Reduction using Importance Masking"""

    def __init__(self, reduction_ratio=0.3):
        self.reduction_ratio = reduction_ratio

    def prune_tokens(self, input_ids, attention_mask=None):
        """Remove tokens from sequence"""
        seq_len = input_ids.shape[1]
        keep_len = int(seq_len * (1 - self.reduction_ratio))

        # Keep first tokens (preserve prompt)
        pruned_ids = input_ids[:, :keep_len]
        pruned_mask = attention_mask[:, :keep_len] if attention_mask is not None else None

        return pruned_ids, pruned_mask

# ============================================================================
# SECTION 5: ACTIVATION QUANTIZATION
# ============================================================================

class ActivationQuantizer:
    """Custom activation quantization"""

    def __init__(self, bits=8):
        self.bits = bits
        self.qmin = -(2**(bits-1))
        self.qmax = 2**(bits-1) - 1
        self.enabled = True

    def quantize(self, x):
        """Quantize activations"""
        if not self.enabled or not isinstance(x, torch.Tensor):
            return x

        x_min = x.min()
        x_max = x.max()

        if x_max == x_min:
            return x

        scale = (x_max - x_min) / (self.qmax - self.qmin)
        zero_point = self.qmin - x_min / scale

        x_quant = torch.round((x / scale) + zero_point)
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
        x_dequant = (x_quant - zero_point) * scale

        return x_dequant

# ============================================================================
# SECTION 6: MODEL CONFIGURATIONS
# ============================================================================

MODELS_CONFIG = {
    # Text models
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "type": "text",
        "min_vram": 3,
        "batch_size": 16
    },
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B-Instruct",
        "type": "text",
        "min_vram": 2,
        "batch_size": 32
    },
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "text",
        "min_vram": 2,
        "batch_size": 32
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "text",
        "min_vram": 8,
        "batch_size": 8
    },
    # Multimodal models
    "qwen2-vl-2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "multimodal",
        "min_vram": 5,
        "batch_size": 4
    },
    "llava-1.5-7b": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "type": "multimodal",
        "min_vram": 14,
        "batch_size": 2
    }
}

# Simplified optimization configs (remove broken AWQ)
OPTIMIZATION_CONFIGS = {
    "fp16_baseline": {
        "name": "FP16 Baseline",
        "dtype": torch.float16,
        "quantization": None,
        "activation_quant": False,
        "trim": False
    },
    "int8_quant": {
        "name": "INT8 Quantization",
        "dtype": torch.float16,
        "quantization": "int8",
        "activation_quant": False,
        "trim": False
    },
    "int4_quant": {
        "name": "INT4 Quantization",
        "dtype": torch.float16,
        "quantization": "int4",
        "activation_quant": False,
        "trim": False
    },
    "int4_activation": {
        "name": "INT4 + Activation Quant",
        "dtype": torch.float16,
        "quantization": "int4",
        "activation_quant": True,
        "trim": False
    },
    "int4_trim": {
        "name": "INT4 + TRIM",
        "dtype": torch.float16,
        "quantization": "int4",
        "activation_quant": False,
        "trim": True
    },
    "int4_full": {
        "name": "INT4 + Act.Quant + TRIM",
        "dtype": torch.float16,
        "quantization": "int4",
        "activation_quant": True,
        "trim": True
    }
}

# ============================================================================
# SECTION 7: IMPROVED DATASETS
# ============================================================================

def create_text_dataset(size=50):
    """Create synthetic text QA dataset with better prompts"""
    data = []
    questions = [
        ("What is the capital of France?", ["Paris", "London", "Berlin", "Madrid"], 0),
        ("Which element has atomic number 6?", ["Carbon", "Oxygen", "Nitrogen", "Hydrogen"], 0),
        ("What year did World War II end?", ["1945", "1944", "1946", "1943"], 0),
        ("What is the largest planet?", ["Jupiter", "Saturn", "Neptune", "Uranus"], 0),
        ("Who wrote Romeo and Juliet?", ["Shakespeare", "Dickens", "Austen", "Bronte"], 0),
        ("What is 2 + 2?", ["4", "5", "3", "6"], 0),
        ("What color is the sky?", ["Blue", "Green", "Red", "Yellow"], 0),
        ("How many days in a week?", ["7", "8", "6", "5"], 0),
        ("Water freezes at?", ["0°C", "32°C", "100°C", "-10°C"], 0),
        ("Who painted Mona Lisa?", ["Da Vinci", "Picasso", "Van Gogh", "Monet"], 0),
    ]

    for i in range(size):
        q, choices, ans = questions[i % len(questions)]
        data.append({
            'question': q,
            'choices': choices,
            'answer': ans,
            'type': 'text'
        })

    return data

def create_visual_dataset(size=30):
    """Create synthetic visual QA dataset"""
    data = []
    colors = [
        ('red', (255, 0, 0)),
        ('blue', (0, 0, 255)),
        ('green', (0, 255, 0)),
        ('yellow', (255, 255, 0)),
        ('purple', (128, 0, 128))
    ]

    for i in range(size):
        color_name, color_rgb = colors[i % len(colors)]
        img = Image.new('RGB', (336, 336), color=color_rgb)

        data.append({
            'image': img,
            'question': f"What color is shown in this image?",
            'choices': ['red', 'blue', 'green', 'yellow', 'purple'],
            'answer': color_name,
            'type': 'visual'
        })

    return data

# ============================================================================
# SECTION 8: IMPROVED MODEL LOADING
# ============================================================================

def get_quantization_config(quant_type):
    """Get quantization configuration"""
    if quant_type == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quant_type == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    return None

def load_model_with_optimization(model_name, optimization):
    """Load model with optimization"""
    model_cfg = MODELS_CONFIG[model_name]
    opt_cfg = OPTIMIZATION_CONFIGS[optimization]

    print(f"  Loading {model_name} with {opt_cfg['name']}...")

    try:
        gc.collect()
        torch.cuda.empty_cache()

        quant_config = None
        if opt_cfg['quantization']:
            quant_config = get_quantization_config(opt_cfg['quantization'])

        model_id = model_cfg['hf_id']

        # Load based on model type
        if model_cfg['type'] == 'multimodal':
            if 'qwen2-vl' in model_name:
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=opt_cfg['dtype'],
                    device_map="auto",
                    quantization_config=quant_config,
                    token=HF_TOKEN,
                    trust_remote_code=True
                )
                processor = Qwen2VLProcessor.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    trust_remote_code=True
                )
            elif 'llava' in model_name:
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=opt_cfg['dtype'],
                    device_map="auto",
                    quantization_config=quant_config,
                    token=HF_TOKEN
                )
                processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        else:  # Text models
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

            # Set pad_token_id in model config
            model.config.pad_token_id = processor.pad_token_id

        # Add optimization modules
        if opt_cfg['activation_quant']:
            model.activation_quantizer = ActivationQuantizer(bits=8)

        if opt_cfg['trim']:
            model.trim = TRIM(reduction_ratio=0.3)

        print(f"  ✓ Loaded successfully")
        return model, processor, "success"

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Failed: {error_msg[:100]}")
        return None, None, error_msg[:100]

# ============================================================================
# SECTION 9: IMPROVED EVALUATION
# ============================================================================

def evaluate_model(model, processor, dataset, model_cfg, opt_cfg, dataset_type):
    """Evaluate model with improved error handling"""

    results = {
        'correct': 0,
        'total': 0,
        'latencies': [],
        'memory_peak': 0,
        'memory_allocated': 0,
        'errors': [],
        'token_throughput': []
    }

    has_trim = hasattr(model, 'trim')
    has_act_quant = hasattr(model, 'activation_quantizer')
    is_multimodal = model_cfg['type'] == 'multimodal'

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    eval_samples = min(30, len(dataset))

    for i, sample in enumerate(tqdm(dataset[:eval_samples], desc=f"  Eval {dataset_type}", leave=False)):
        try:
            # Prepare inputs
            if dataset_type == 'text' or sample.get('type') == 'text':
                question = sample['question']
                choices = sample['choices']
                answer_idx = sample['answer']

                # Improved prompt
                prompt = f"Answer with only the letter (A, B, C, or D).\n\nQuestion: {question}\n"
                for j, choice in enumerate(choices):
                    prompt += f"{chr(65+j)}. {choice}\n"
                prompt += "\nAnswer:"

                inputs = processor(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

            else:  # Visual
                image = sample.get('image')
                question = sample.get('question', 'What is this?')

                if image is None:
                    continue

                if 'qwen2-vl' in model_cfg['hf_id']:
                    # Qwen2-VL format
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"Answer with one word only. {question}"}
                        ]
                    }]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True
                    )
                else:  # LLaVA
                    prompt = f"USER: <image>\n{question} Answer in one word.\nASSISTANT:"
                    inputs = processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )

                inputs = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in inputs.items()}

            # Apply TRIM if enabled
            if has_trim and 'input_ids' in inputs:
                inputs['input_ids'], inputs['attention_mask'] = model.trim.prune_tokens(
                    inputs['input_ids'],
                    inputs.get('attention_mask')
                )

            # Measure inference
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                if has_act_quant:
                    hooks = []
                    def quant_hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            return model.activation_quantizer.quantize(output)
                        return output

                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            hooks.append(layer.register_forward_hook(quant_hook))

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        temperature=None,
                        top_p=None
                    )

                    for hook in hooks:
                        hook.remove()
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        temperature=None,
                        top_p=None
                    )

            torch.cuda.synchronize()
            latency = time.time() - start_time
            results['latencies'].append(latency)

            # Calculate throughput
            output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 5
            tokens_per_sec = output_tokens / latency if latency > 0 else 0
            results['token_throughput'].append(tokens_per_sec)

            # Memory tracking
            current_mem = torch.cuda.memory_allocated() / 1e9
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            results['memory_allocated'] = max(results['memory_allocated'], current_mem)
            results['memory_peak'] = max(results['memory_peak'], peak_mem)

            # Check answer
            if dataset_type == 'text' or sample.get('type') == 'text':
                if 'input_ids' in inputs:
                    response = processor.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                else:
                    response = processor.decode(outputs[0], skip_special_tokens=True).strip()

                # Extract letter
                response_upper = response.upper()
                pred_letter = None
                for char in response_upper:
                    if char in 'ABCD':
                        pred_letter = char
                        break

                if pred_letter:
                    pred_idx = ord(pred_letter) - ord('A')
                    if pred_idx == answer_idx:
                        results['correct'] += 1
            else:  # Visual
                response = processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
                expected = sample['answer'].lower()

                # Check if answer color is in response
                if expected in response:
                    results['correct'] += 1

            results['total'] += 1

        except Exception as e:
            error_msg = str(e)[:50]
            results['errors'].append(error_msg)
            continue

    # Calculate metrics
    if results['total'] > 0:
        return {
            'accuracy': (results['correct'] / results['total']) * 100,
            'avg_latency_ms': np.mean(results['latencies']) * 1000,
            'std_latency_ms': np.std(results['latencies']) * 1000,
            'min_latency_ms': np.min(results['latencies']) * 1000,
            'max_latency_ms': np.max(results['latencies']) * 1000,
            'p50_latency_ms': np.percentile(results['latencies'], 50) * 1000,
            'p95_latency_ms': np.percentile(results['latencies'], 95) * 1000,
            'p99_latency_ms': np.percentile(results['latencies'], 99) * 1000,
            'avg_tokens_per_sec': np.mean(results['token_throughput']),
            'memory_allocated_gb': results['memory_allocated'],
            'memory_peak_gb': results['memory_peak'],
            'samples_evaluated': results['total'],
            'samples_correct': results['correct'],
            'error_count': len(results['errors']),
            'error_rate': len(results['errors']) / (results['total'] + len(results['errors'])) * 100
        }

    return None

# ============================================================================
# SECTION 10: MAIN EVALUATION LOOP
# ============================================================================

print("\n" + "=" * 80)
print("STARTING COMPREHENSIVE EVALUATION")
print("=" * 80)

# Store results
results_data = []
skipped_configs = []
failed_loads = []

# Create datasets
print("\nCreating test datasets...")
text_dataset = create_text_dataset(size=50)
visual_dataset = create_visual_dataset(size=30)
print(f"✓ Created {len(text_dataset)} text samples")
print(f"✓ Created {len(visual_dataset)} visual samples")

# Determine models to test
models_to_test = list(MODELS_CONFIG.keys())
if gpu_memory < 10:
    models_to_test = [m for m in models_to_test if MODELS_CONFIG[m]['min_vram'] < 10]
    print(f"\n⚠ Limited VRAM - testing smaller models only")

optimizations_to_test = list(OPTIMIZATION_CONFIGS.keys())

print(f"\nModels to test: {models_to_test}")
print(f"Optimizations: {len(optimizations_to_test)}")
print(f"Total configurations: {len(models_to_test) * len(optimizations_to_test)}")

# Main loop
total_configs = len(models_to_test) * len(optimizations_to_test)
completed = 0

for model_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({MODELS_CONFIG[model_name]['type'].upper()})")
    print(f"{'='*70}")

    model_cfg = MODELS_CONFIG[model_name]

    for opt_name in optimizations_to_test:
        opt_cfg = OPTIMIZATION_CONFIGS[opt_name]
        completed += 1

        print(f"\n[{completed}/{total_configs}] {opt_cfg['name']}")

        # Load model
        model, processor, status = load_model_with_optimization(model_name, opt_name)

        if model is None:
            failed_loads.append({
                'model': model_name,
                'optimization': opt_cfg['name'],
                'error': status
            })
            continue

        # Determine datasets
        datasets_to_eval = []
        if model_cfg['type'] == 'text':
            datasets_to_eval = [('text', text_dataset)]
        elif model_cfg['type'] == 'multimodal':
            datasets_to_eval = [('text', text_dataset), ('visual', visual_dataset)]

        # Evaluate
        for dataset_type, dataset in datasets_to_eval:
            metrics = evaluate_model(
                model, processor, dataset,
                model_cfg, opt_cfg, dataset_type
            )

            if metrics:
                result = {
                    'model': model_name,
                    'model_type': model_cfg['type'],
                    'optimization': opt_cfg['name'],
                    'dataset_type': dataset_type,
                    'accuracy': metrics['accuracy'],
                    'avg_latency_ms': metrics['avg_latency_ms'],
                    'std_latency_ms': metrics['std_latency_ms'],
                    'p50_latency_ms': metrics['p50_latency_ms'],
                    'p95_latency_ms': metrics['p95_latency_ms'],
                    'p99_latency_ms': metrics['p99_latency_ms'],
                    'min_latency_ms': metrics['min_latency_ms'],
                    'max_latency_ms': metrics['max_latency_ms'],
                    'tokens_per_sec': metrics['avg_tokens_per_sec'],
                    'memory_allocated_gb': metrics['memory_allocated_gb'],
                    'memory_peak_gb': metrics['memory_peak_gb'],
                    'samples_evaluated': metrics['samples_evaluated'],
                    'samples_correct': metrics['samples_correct'],
                    'error_count': metrics['error_count'],
                    'error_rate': metrics['error_rate'],
                    'timestamp': datetime.now().isoformat()
                }
                results_data.append(result)

                print(f"    ✓ Accuracy: {metrics['accuracy']:.1f}%")
                print(f"    ✓ Latency: {metrics['avg_latency_ms']:.1f}ms (±{metrics['std_latency_ms']:.1f}ms)")
                print(f"    ✓ Throughput: {metrics['avg_tokens_per_sec']:.1f} tokens/sec")
                print(f"    ✓ Memory: {metrics['memory_peak_gb']:.2f}GB peak")

        # Cleanup
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 11: COMPREHENSIVE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("=" * 80)

if results_data:
    df = pd.DataFrame(results_data)

    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    print(f"Total Experiments Run: {len(results_data)}")
    print(f"Models Tested: {df['model'].nunique()}")
    print(f"Optimization Methods: {df['optimization'].nunique()}")
    print(f"Failed Loads: {len(failed_loads)}")
    print(f"Success Rate: {len(results_data)/(len(results_data)+len(failed_loads))*100:.1f}%")

    print("\n" + "="*70)
    print("COMPLETE RESULTS TABLE")
    print("="*70)
    display_cols = ['model', 'optimization', 'dataset_type', 'accuracy',
                    'avg_latency_ms', 'tokens_per_sec', 'memory_peak_gb']
    print(df[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("PERFORMANCE BY OPTIMIZATION METHOD")
    print("="*70)
    opt_stats = df.groupby('optimization').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'avg_latency_ms': ['mean', 'std'],
        'tokens_per_sec': ['mean', 'std'],
        'memory_peak_gb': ['mean', 'std'],
        'error_rate': ['mean', 'max']
    }).round(2)
    print(opt_stats)

    print("\n" + "="*70)
    print("PERFORMANCE BY MODEL")
    print("="*70)
    model_stats = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'avg_latency_ms': ['mean', 'std'],
        'tokens_per_sec': ['mean'],
        'memory_peak_gb': ['mean', 'max']
    }).round(2)
    print(model_stats)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS BY ACCURACY")
    print("="*70)
    top_accuracy = df.nlargest(10, 'accuracy')[display_cols]
    print(top_accuracy.to_string(index=False))

    print("\n" + "="*70)
    print("TOP 10 BY MEMORY EFFICIENCY")
    print("="*70)
    top_memory = df.nsmallest(10, 'memory_peak_gb')[
        ['model', 'optimization', 'dataset_type', 'memory_peak_gb', 'accuracy']
    ]
    print(top_memory.to_string(index=False))

    print("\n" + "="*70)
    print("COMPRESSION EFFICIENCY ANALYSIS")
    print("="*70)
    baseline = df[df['optimization'] == 'FP16 Baseline']

    if not baseline.empty:
        for model in df['model'].unique():
            model_baseline = baseline[baseline['model'] == model]
            model_data = df[df['model'] == model]

            if model_baseline.empty:
                continue

            print(f"\n--- {model.upper()} ---")

            baseline_acc = model_baseline['accuracy'].mean()
            baseline_lat = model_baseline['avg_latency_ms'].mean()
            baseline_mem = model_baseline['memory_peak_gb'].mean()

            print(f"Baseline: {baseline_acc:.1f}% acc, {baseline_lat:.1f}ms, {baseline_mem:.2f}GB")

            for opt in model_data['optimization'].unique():
                if opt == 'FP16 Baseline':
                    continue

                opt_data = model_data[model_data['optimization'] == opt]
                if opt_data.empty:
                    continue

                opt_acc = opt_data['accuracy'].mean()
                opt_lat = opt_data['avg_latency_ms'].mean()
                opt_mem = opt_data['memory_peak_gb'].mean()

                mem_reduction = (1 - opt_mem / baseline_mem) * 100 if baseline_mem > 0 else 0
                speed_change = ((opt_lat / baseline_lat) - 1) * 100 if baseline_lat > 0 else 0
                acc_change = opt_acc - baseline_acc

                print(f"\n{opt}:")
                print(f"  Accuracy: {opt_acc:.1f}% ({acc_change:+.1f}%)")
                print(f"  Latency: {opt_lat:.1f}ms ({speed_change:+.1f}%)")
                print(f"  Memory: {opt_mem:.2f}GB ({mem_reduction:+.1f}% reduction)")

    if failed_loads:
        print("\n" + "="*70)
        print("FAILED LOADS")
        print("="*70)
        failed_df = pd.DataFrame(failed_loads)
        print(failed_df.to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print("\nBest for ACCURACY:")
    best_acc = df.loc[df['accuracy'].idxmax()]
    print(f"  {best_acc['model']} + {best_acc['optimization']}")
    print(f"  Accuracy: {best_acc['accuracy']:.1f}%")

    print("\nBest for MEMORY:")
    best_mem = df.loc[df['memory_peak_gb'].idxmin()]
    print(f"  {best_mem['model']} + {best_mem['optimization']}")
    print(f"  Memory: {best_mem['memory_peak_gb']:.2f}GB")

    print("\n" + "="*70)
    print("DATA AVAILABLE IN MEMORY")
    print("="*70)
    print(f"✓ results_data: List with {len(results_data)} results")
    print(f"✓ df: DataFrame with shape {df.shape}")
    print(f"✓ failed_loads: {len(failed_loads)} failed configs")

else:
    print("\n⚠ No results collected")

print("\n" + "=" * 80)
print("✓ COMPREHENSIVE EVALUATION COMPLETE")
print("=" * 80)
