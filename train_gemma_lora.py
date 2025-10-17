#!/usr/bin/env python3
"""
Train Gemma Vision Model with LoRA on Zipper Defect Classification
- No quantization (full precision training)
- Uses LoRA for efficient fine-tuning
- Handles both old (string) and new (list) dataset formats
- Saves to HuggingFace Hub
"""
import os
import json
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).parent

def parse_args():
    parser = argparse.ArgumentParser(description="Train Gemma model with LoRA")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it",
                        help="Base model ID (default: google/gemma-3-4b-it)")
    parser.add_argument("--processor_id", type=str, default="google/gemma-3-4b-it",
                        help="Processor ID (default: google/gemma-3-4b-it)")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="gemma-zipper-lora",
                        help="Output directory (default: gemma-zipper-lora)")
    parser.add_argument("--hub_repo", type=str, default="ayushadarsh7/gemma3_lora",
                        help="HuggingFace Hub repository name (e.g., username/model-name)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA r parameter (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter (default: 16)")
    parser.add_argument("--merge_and_save", action="store_true",
                        help="Merge LoRA adapter with base model and save")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training to N samples (for testing)")
    return parser.parse_args()

# -----------------------------
# Dataset Loading
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

def load_jsonl_dataset(jsonl_path, max_samples=None):
    """
    Load dataset from JSONL file.
    Handles both old (string) and new (list) formats.
    """
    data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                entry = json.loads(line.strip())
                
                # Get user content
                user_content = entry['messages'][0]['content']
                
                # Check format and convert to standard format
                if isinstance(user_content, list):
                    # Already in Gemma format - use as-is
                    data.append(entry)
                else:
                    # Old string format - convert to Gemma format
                    import re
                    image_url_match = re.search(r'\[Image:\s*(.+?)\]', user_content)
                    if not image_url_match:
                        print(f"⚠️  Skipping entry {idx+1}: No image URL found")
                        continue
                    
                    image_url = image_url_match.group(1).strip()
                    prompt = user_content.split('[Image:')[0].strip()
                    
                    # Convert to Gemma format
                    converted_entry = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "url": image_url},
                                    {"type": "text", "text": prompt}
                                ]
                            },
                            entry['messages'][1]  # Keep assistant message as-is
                        ]
                    }
                    data.append(converted_entry)
                    
            except Exception as e:
                print(f"⚠️  Error processing line {idx+1}: {e}")
                continue
    
    return data

def process_vision_info(messages, download_images=True):
    """
    Extract images from messages.
    If download_images=True, downloads images from URLs.
    """
    image_inputs = []
    
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        
        for element in content:
            if isinstance(element, dict):
                # Check for image
                if element.get("type") == "image":
                    if "url" in element and download_images:
                        # Download image from URL
                        image = load_image_from_url(element["url"])
                        if image:
                            image_inputs.append(image)
                    elif "image" in element:
                        # Direct PIL image
                        image = element["image"]
                        if isinstance(image, Image.Image):
                            image_inputs.append(image.convert("RGB"))
    
    return image_inputs

# -----------------------------
# Main Training Function
# -----------------------------
def main():
    args = parse_args()
    
    # Load HuggingFace token
    load_dotenv()
    hf_token = os.getenv("hf_token")
    
    if not hf_token:
        print("⚠️  Warning: No hf_token found in .env file")
        print("   You may not be able to access gated models or push to hub")
    else:
        print("🔐 Logging in to HuggingFace...")
        login(token=hf_token)
        print("✅ HuggingFace login successful")
    
    print("\n" + "="*70)
    print("🚀 GEMMA LORA TRAINING - ZIPPER DEFECT CLASSIFICATION")
    print("="*70)
    
    # Check GPU and precision
    if not torch.cuda.is_available():
        raise ValueError("❌ CUDA not available. This script requires a GPU.")
    
    # Check GPU capability
    gpu_capability = torch.cuda.get_device_capability()[0]
    print(f"\n🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {gpu_capability}.{torch.cuda.get_device_capability()[1]}")
    
    # Use FP32 for training (full precision)
    dtype = torch.float32
    precision_name = "FP32"
    
    print(f"   Training Precision: {precision_name} (Full Precision)")
    print(f"   No Quantization")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.train_jsonl}...")
    dataset = load_jsonl_dataset(args.train_jsonl, max_samples=args.max_samples)
    
    if not dataset:
        raise ValueError("❌ No data loaded. Check your JSONL file.")
    
    print(f"✅ Loaded {len(dataset)} training samples")
    
    # Show sample
    print(f"\n📝 Sample training entry:")
    print(json.dumps(dataset[0], indent=2, default=str)[:500] + "...")
    
    # Load model and processor
    print(f"\n🔧 Loading model and processor...")
    print(f"   Base model: {args.model_id}")
    print(f"   Processor: {args.processor_id}")
    
    model_kwargs = dict(
        attn_implementation="eager",  # Use "flash_attention_2" for Ampere+ GPUs
        torch_dtype=dtype,
        device_map="auto",
    )
    
    # Load model WITHOUT quantization
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.processor_id)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model type: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Configure LoRA
    print(f"\n🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    
    print(f"   LoRA r: {args.lora_r}")
    print(f"   LoRA alpha: {args.lora_alpha}")
    print(f"   Target modules: all-linear")
    
    # Configure training arguments
    output_dir = SCRIPT_DIR / args.output_dir
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        bf16=False,  # Use FP32
        fp16=False,  # Use FP32
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",  # dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    
    print(f"\n📊 Training Configuration:")
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Push to hub: {bool(args.hub_repo)}")
    if args.hub_repo:
        print(f"   Hub repo: {args.hub_repo}")
    
    # Create data collator
    def collate_fn(examples):
        texts = []
        images = []
        
        for example in examples:
            # Download images from URLs
            image_inputs = process_vision_info(example["messages"], download_images=True)
            
            # Apply chat template
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            
            texts.append(text.strip())
            images.append(image_inputs)
        
        # Tokenize texts and process images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # Create labels (mask padding and image tokens)
        labels = batch["input_ids"].clone()
        
        # Get image token ID
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
        
        # Mask tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100  # Additional image token
        
        batch["labels"] = labels
        return batch
    
    # Initialize trainer
    print(f"\n🔧 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    # Start training
    print(f"\n{'='*70}")
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    trainer.train()
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print("="*70)
    
    # Save model
    print(f"\n💾 Saving model...")
    trainer.save_model()
    print(f"✅ Model saved to {output_dir}")
    
    # Push to hub if specified
    if args.hub_repo:
        print(f"\n📤 Pushing to HuggingFace Hub: {args.hub_repo}")
        trainer.push_to_hub()
        print(f"✅ Model pushed to hub successfully")
    
    # Merge and save full model if requested
    if args.merge_and_save:
        print(f"\n🔧 Merging LoRA adapter with base model...")
        
        # Free memory
        del model
        del trainer
        torch.cuda.empty_cache()
        
        # Load base model
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, 
            low_cpu_mem_usage=True,
            torch_dtype=dtype
        )
        
        # Load and merge LoRA
        peft_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model
        merged_dir = output_dir / "merged_model"
        merged_model.save_pretrained(
            str(merged_dir), 
            safe_serialization=True, 
            max_shard_size="2GB"
        )
        processor.save_pretrained(str(merged_dir))
        
        print(f"✅ Merged model saved to {merged_dir}")
        
        # Push merged model to hub if specified
        if args.hub_repo:
            print(f"\n📤 Pushing merged model to hub: {args.hub_repo}-merged")
            merged_model.push_to_hub(f"{args.hub_repo}-merged")
            processor.push_to_hub(f"{args.hub_repo}-merged")
            print(f"✅ Merged model pushed to hub")
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE!")
    print("="*70)
    print(f"\n📁 Model location: {output_dir}")
    if args.hub_repo:
        print(f"🤗 HuggingFace Hub: https://huggingface.co/{args.hub_repo}")
    print(f"\n💡 To use the model:")
    print(f"   python gemma_inference.py --model_id {output_dir}")
    if args.hub_repo:
        print(f"   Or: python gemma_inference.py --model_id {args.hub_repo}")

if __name__ == "__main__":
    main()

