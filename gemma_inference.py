#!/usr/bin/env python3
"""
Gemma Model Inference Script for Fine-tuned Models
Runs inference on JSONL dataset and generates classification report
Supports GPU acceleration
"""
import os
import json
import argparse
import re
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel, PeftConfig

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).parent

# Class taxonomy
CLASSES = [
    'Fabric Tear near zipper',
    'No evidence for reporting any significant zip issue',
    'Zip pull tab broken or missing or detached',
    'Zip slider completely off track',
    'Zip slider not interlocking zip',
    'Zip slider off track from one side',
    'Zip teeth damaged'
]

# -----------------------------
# Parse Arguments
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gemma model inference on JSONL dataset")
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID (e.g., 'google/gemma-3-4b-it') or path to fine-tuned model")
    parser.add_argument("--base_model_id", type=str, default=None,
                        help="Base model ID if loading LoRA adapter (e.g., 'google/gemma-3-4b-pt')")
    parser.add_argument("--is_lora", action="store_true",
                        help="Load model as LoRA adapter on top of base model")
    parser.add_argument("--jsonl", type=str, default="gemma_test_dataset_direct.jsonl",
                        help="Input JSONL file (default: gemma_test_dataset_direct.jsonl)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit inference to first N samples (for testing)")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Output directory for results (default: inference_results)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation (default: 0.0 for deterministic)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu, default: auto-detect)")
    return parser.parse_args()

# -----------------------------
# Dataset Class
# -----------------------------
class GemmaInferenceDataset(Dataset):
    """Dataset for loading JSONL data for inference"""
    
    def __init__(self, jsonl_path, n_samples=None):
        self.data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if n_samples and idx >= n_samples:
                    break
                try:
                    entry = json.loads(line.strip())
                    self.data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {idx+1}: Invalid JSON - {e}")
        
        print(f"✅ Loaded {len(self.data)} samples from {Path(jsonl_path).name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Extract user message content
        user_content = entry['messages'][0]['content']
        
        # Extract true label from assistant message
        true_label = entry['messages'][1]['content']
        
        # Check if content is already in Gemma format (list) or old format (string)
        if isinstance(user_content, list):
            # New Gemma format: content is a list with image and text dicts
            image_url = None
            prompt = None
            
            for item in user_content:
                if item.get('type') == 'image':
                    image_url = item.get('url')
                elif item.get('type') == 'text':
                    prompt = item.get('text')
        else:
            # Old format: content is a string with [Image: <url>]
            image_url_match = re.search(r'\[Image:\s*(.+?)\]', user_content)
            image_url = image_url_match.group(1).strip() if image_url_match else None
            prompt = user_content.split('[Image:')[0].strip()
        
        return {
            'image_url': image_url,
            'prompt': prompt,
            'true_label': true_label,
            'messages': entry['messages']  # Store full messages for Gemma format
        }

# -----------------------------
# Helper Functions
# -----------------------------
def load_image_from_url(url, timeout=30):
    """Download and load image from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"❌ Error loading image from {url}: {e}")
        return None

def extract_class_name(prediction_text, valid_classes=CLASSES):
    """
    Extract the class name from model output.
    Handles cases where model outputs extra text.
    """
    prediction_text = prediction_text.strip()
    
    # Try exact match first
    if prediction_text in valid_classes:
        return prediction_text
    
    # Try case-insensitive match
    for cls in valid_classes:
        if prediction_text.lower() == cls.lower():
            return cls
    
    # Try partial match (class name contained in output)
    for cls in valid_classes:
        if cls.lower() in prediction_text.lower():
            return cls
    
    # If no match, return the raw prediction
    return prediction_text

# -----------------------------
# Inference Function
# -----------------------------
def run_inference(model, processor, dataset, device, temperature=0.0, max_new_tokens=100, batch_size=1):
    """Run inference on the dataset"""
    
    model.eval()
    predictions = []
    true_labels = []
    image_urls = []
    errors = []
    
    # Use DataLoader for potential batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING INFERENCE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*70}\n")
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            batch_predictions = []
            
            for i in range(len(batch['image_url'])):
                image_url = batch['image_url'][i]
                prompt = batch['prompt'][i]
                true_label = batch['true_label'][i]
                
                try:
                    # Construct Gemma format messages from extracted fields
                    # This avoids issues with DataLoader batching complex nested structures
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "url": image_url},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(device)
                    
                    # Generate prediction
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                    
                    # Decode output (skip input tokens)
                    prediction_text = processor.decode(
                        outputs[0][inputs["input_ids"].shape[-1]:],
                        skip_special_tokens=True
                    )
                    
                    # Extract class name
                    predicted_class = extract_class_name(prediction_text)
                    batch_predictions.append(predicted_class)
                    
                except Exception as e:
                    import traceback
                    print(f"\n❌ Error processing sample:")
                    print(f"   URL: {image_url}")
                    print(f"   Error: {e}")
                    print(f"   Traceback: {traceback.format_exc()}")
                    batch_predictions.append("ERROR_PROCESSING")
                    errors.append({
                        'url': image_url,
                        'error': str(e)
                    })
                
                # Collect data
                true_labels.append(true_label)
                image_urls.append(image_url)
            
            predictions.extend(batch_predictions)
    
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors during inference")
    
    return predictions, true_labels, image_urls, errors

# -----------------------------
# Report Generation
# -----------------------------
def generate_classification_report(predictions, true_labels, image_urls, errors, 
                                   model_name, output_dir, n_samples, total_in_file):
    """Generate comprehensive classification report"""
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_url': image_urls,
        'true_label': true_labels,
        'predicted_label': predictions
    })
    
    # Save predictions CSV
    os.makedirs(output_dir, exist_ok=True)
    limit_suffix = f"_first{n_samples}" if n_samples else ""
    output_csv = output_dir / f"{model_name}_predictions_{len(df)}samples{limit_suffix}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")
    
    # Filter out errors
    valid_idx = ~df['predicted_label'].str.startswith('ERROR')
    errors_count = len(df) - valid_idx.sum()
    
    if errors_count > 0:
        print(f"\n⚠️  Found {errors_count} errors during prediction")
    
    y_true = df.loc[valid_idx, 'true_label']
    y_pred = df.loc[valid_idx, 'predicted_label']
    
    # Calculate metrics
    overall_acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    
    # Generate report file
    report_file = output_dir / f"{model_name}_classification_report_{len(df)}samples{limit_suffix}.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNED GEMMA MODEL CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Total samples processed: {len(df)}\n")
        if n_samples:
            f.write(f"⚠️  LIMITED RUN: First {n_samples} of {total_in_file} total samples in file\n")
        f.write(f"Errors during prediction: {errors_count}\n")
        f.write(f"Valid predictions: {valid_idx.sum()}\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
        f.write("=" * 80 + "\n")
        f.write("\nPER-CLASS METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, cls in enumerate(CLASSES):
            true_pos = sum((y_true == cls) & (y_pred == cls))
            total_true = sum(y_true == cls)
            acc = true_pos / total_true if total_true > 0 else 0.0
            
            f.write(f"Class: {cls}\n")
            f.write(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall:    {recall[i]:.4f}\n")
            f.write(f"  F1-Score:  {f1[i]:.4f}\n")
            f.write(f"  Support:   {support[i]}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
        
        # Error analysis
        misclassified = df.loc[valid_idx & (df['true_label'] != df['predicted_label'])]
        f.write("\n" + "=" * 80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total misclassifications: {len(misclassified)} out of {valid_idx.sum()} valid predictions\n")
        f.write(f"Error rate: {len(misclassified)/valid_idx.sum()*100:.2f}%\n\n")
        
        if len(misclassified) > 0:
            f.write("Sample misclassifications (first 10):\n\n")
            for idx, (_, row) in enumerate(misclassified.head(10).iterrows(), 1):
                f.write(f"{idx}. Image: {row['image_url']}\n")
                f.write(f"   True label: {row['true_label']}\n")
                f.write(f"   Predicted:  {row['predicted_label']}\n\n")
        
        # Prediction errors (technical failures)
        if errors_count > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TECHNICAL ERRORS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total technical errors: {errors_count}\n\n")
            error_samples = df[~valid_idx].head(10)
            for idx, (_, row) in enumerate(error_samples.iterrows(), 1):
                f.write(f"{idx}. Image: {row['image_url']}\n")
                f.write(f"   Error type: {row['predicted_label']}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION PATTERNS\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze top confusion pairs
        confusion_counts = Counter()
        for _, row in misclassified.iterrows():
            confusion_counts[(row['true_label'], row['predicted_label'])] += 1
        
        if confusion_counts:
            f.write("Top 10 confusion pairs (True -> Predicted):\n\n")
            for (true_cls, pred_cls), count in confusion_counts.most_common(10):
                f.write(f"  {count}x  {true_cls}\n")
                f.write(f"      -> {pred_cls}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Valid predictions: {valid_idx.sum()}\n")
        f.write(f"Correct predictions: {valid_idx.sum() - len(misclassified)}\n")
        f.write(f"Incorrect predictions: {len(misclassified)}\n")
        f.write(f"Technical errors: {errors_count}\n")
        f.write(f"Model: {model_name}\n")
    
    print(f"✅ Classification report saved to {report_file}")
    
    # Print summary to console
    print(f"\n{'='*70}")
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Correct: {valid_idx.sum() - len(misclassified)}/{valid_idx.sum()}")
    print(f"Incorrect: {len(misclassified)}/{valid_idx.sum()}")
    print(f"Technical errors: {errors_count}")
    
    if len(misclassified) > 0:
        print(f"\n⚠️  Misclassified samples: {len(misclassified)}")
        print("\nTop 3 misclassification examples:")
        for idx, (_, row) in enumerate(misclassified.head(3).iterrows(), 1):
            print(f"\n  {idx}. True: {row['true_label']}")
            print(f"     Pred: {row['predicted_label']}")
            print(f"     URL:  {row['image_url'][:80]}...")
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved in: {output_dir}/")
    print("="*70)

# -----------------------------
# Main Function
# -----------------------------
def main():
    args = parse_args()
    
    # Load HuggingFace token and login
    load_dotenv()
    hf_token = os.getenv("hf_token")
    
    if hf_token:
        print("🔐 Logging in to HuggingFace...")
        login(token=hf_token)
        print("✅ HuggingFace login successful")
    else:
        print("⚠️  No HF_TOKEN found in .env - proceeding without authentication")
        print("   (This may fail for gated models)")
    
    # Resolve paths
    jsonl_path = SCRIPT_DIR / args.jsonl if not os.path.isabs(args.jsonl) else Path(args.jsonl)
    output_dir = SCRIPT_DIR / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != device:
        print(f"⚠️  Requested {args.device} but using {device}")
    
    print("\n" + "="*70)
    print("🚀 GEMMA MODEL INFERENCE")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  JSONL file: {jsonl_path.name}")
    print(f"  Device: {device.upper()}")
    print(f"  Precision: FP32 (Full Precision)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    if args.n_samples:
        print(f"  Sample limit: {args.n_samples}")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    
    print(f"\n🔧 Loading model and processor...")
    
    # Load model and processor
    try:
        # Use FP32 for inference (full precision)
        dtype = torch.float32
        
        # Check if this is a LoRA checkpoint
        is_lora_checkpoint = args.is_lora
        
        # Auto-detect LoRA checkpoint if adapter_config.json exists
        if not is_lora_checkpoint:
            adapter_config_path = Path(args.model_id) / "adapter_config.json"
            if adapter_config_path.exists():
                print(f"🔍 Detected LoRA adapter at {args.model_id}")
                is_lora_checkpoint = True
        
        if is_lora_checkpoint:
            # Load LoRA adapter
            print(f"🔧 Loading LoRA adapter from {args.model_id}")
            
            # Determine base model
            if args.base_model_id:
                base_model_id = args.base_model_id
            else:
                # Try to read from adapter config
                try:
                    peft_config = PeftConfig.from_pretrained(args.model_id)
                    base_model_id = peft_config.base_model_name_or_path
                    print(f"   Base model from config: {base_model_id}")
                except:
                    raise ValueError(
                        "❌ Could not determine base model. Please specify --base_model_id "
                        "(e.g., --base_model_id google/gemma-3-4b-pt)"
                    )
            
            # Load base model
            print(f"   Loading base model: {base_model_id}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter
            print(f"   Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, args.model_id)
            model = model.merge_and_unload()  # Merge adapter with base model
            model.to(device)
            
            # Load processor from adapter or base model
            try:
                processor = AutoProcessor.from_pretrained(args.model_id)
            except:
                print(f"   Processor not found in adapter, using base model processor")
                processor = AutoProcessor.from_pretrained(base_model_id.replace('-pt', '-it'))
            
            print(f"✅ LoRA model loaded and merged successfully")
        else:
            # Load regular model
            print(f"🔧 Loading full model from {args.model_id}")
            processor = AutoProcessor.from_pretrained(args.model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                torch_dtype=dtype
            )
            model.to(device)
            print(f"✅ Model loaded successfully")
        
        print(f"   Model type: {model.__class__.__name__}")
        print(f"   Device: {device}")
        print(f"   Dtype: {dtype} (Full Precision)")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        import traceback
        print(f"❌ Error loading model:")
        print(f"   {e}")
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        return
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    
    # First count total samples
    total_samples = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_samples += 1
    
    dataset = GemmaInferenceDataset(jsonl_path, n_samples=args.n_samples)
    
    if args.n_samples:
        print(f"⚠️  Processing first {len(dataset)} of {total_samples} samples")
    
    # Run inference
    predictions, true_labels, image_urls, errors = run_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        device=device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    # Generate report
    model_name = Path(args.model_id).name if os.path.isdir(args.model_id) else args.model_id.replace('/', '_')
    generate_classification_report(
        predictions=predictions,
        true_labels=true_labels,
        image_urls=image_urls,
        errors=errors,
        model_name=model_name,
        output_dir=output_dir,
        n_samples=args.n_samples,
        total_in_file=total_samples
    )

if __name__ == "__main__":
    main()

