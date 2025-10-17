#!/usr/bin/env python3
"""
Train Gemma Vision Model - Fine-tune Vision Encoder Layers Only
- Freezes all model parameters
- Unfreezes only vision encoder layers
- Uses FP32 precision
- Handles Gemma format dataset (image URLs in structured messages)
- Saves to local checkpoint and HuggingFace Hub
"""
import os
import json
import argparse
from pathlib import Path
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).parent

def parse_args():
    parser = argparse.ArgumentParser(description="Train Gemma vision encoder layers")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it",
                        help="Base model ID (default: google/gemma-3-4b-it)")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="gemma-vision-checkpoint_lr1e_5_epochs5",
                        help="Output directory (default: gemma-vision-checkpoint)")
    parser.add_argument("--hub_repo", type=str, default="ayushadarsh7/gemma_vision_finetuned_350_param",
                        help="HuggingFace Hub repository name (e.g., username/model-name)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
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
    Load dataset from JSONL file in Gemma format.
    Expected format:
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image", "url": "https://..."},
            {"type": "text", "text": "prompt..."}
          ]
        },
        {
          "role": "assistant",
          "content": "label"
        }
      ]
    }
    """
    data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            try:
                entry = json.loads(line.strip())
                data.append(entry)
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
    print("🚀 GEMMA VISION ENCODER TRAINING")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise ValueError("❌ CUDA not available. This script requires a GPU.")
    
    gpu_capability = torch.cuda.get_device_capability()[0]
    print(f"\n🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute Capability: {gpu_capability}.{torch.cuda.get_device_capability()[1]}")
    
    # Use FP32 for training
    dtype = torch.float32
    print(f"   Training Precision: FP32 (Full Precision)")
    print(f"   Training Strategy: Vision Encoder Layers Only")
    
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
    print(f"   Model: {args.model_id}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    print(f"✅ Model loaded successfully")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params / 1e9:.2f}B")
    
    # Freeze all parameters first
    print(f"\n❄️  Freezing all model parameters...")
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze only vision encoder layers
    print(f"🔓 Unfreezing vision encoder layers...")
    unfrozen_count = 0
    unfrozen_names = []
    
    for name, param in model.named_parameters():
        if "vision_tower.vision_model.encoder" in name:
            param.requires_grad = True
            unfrozen_count += 1
            unfrozen_names.append(name)
    
    # Calculate trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"\n✅ Parameter Freeze Summary:")
    print(f"   Trainable parameters: {trainable / 1e6:.2f}M ({trainable / total_params * 100:.2f}%)")
    print(f"   Frozen parameters: {frozen / 1e6:.2f}M ({frozen / total_params * 100:.2f}%)")
    print(f"   Unfrozen layers: {unfrozen_count}")
    
    if unfrozen_count > 0:
        print(f"\n📋 Sample unfrozen layers:")
        for name in unfrozen_names[:3]:
            print(f"   - {name}")
        if len(unfrozen_names) > 3:
            print(f"   ... and {len(unfrozen_names) - 3} more")
    else:
        print("\n⚠️  WARNING: No layers were unfrozen! Check layer names.")
        print("   Searching for available layer names...")
        for name, _ in list(model.named_parameters())[:10]:
            print(f"   - {name}")
    
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
