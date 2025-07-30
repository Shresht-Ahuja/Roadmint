from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def generate_gemma_roadmap(skill):
    """Generate roadmap with Gemma-2B using CPU offloading"""
    try:
        # Configure 4-bit quantization with CPU offloading
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True  # Critical for low VRAM
        )

        # Custom device map for offloading
        device_map = {
            "model": 0,          # GPU
            "lm_head": 0,        # GPU
            "embed_tokens": 0,   # GPU
            "layers": 0,         # GPU
            "norm": "cpu",       # Offload to CPU
            "final_layer_norm": "cpu"
        }

        # Load model with quantization and offloading
        model_id = "google/gemma-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )

        # Optimized prompt
        prompt = f"""<start_of_turn>user
Generate a structured learning roadmap for {skill} with:
1. Numbered steps (1., 2., etc.)
2. Time estimates ("Time: X weeks")
3. Resource links ("Link: example.com")

Example:
1. Introduction
Learn basic syntax and concepts
Time: 2 weeks
Link: example.com/intro<end_of_turn>
<start_of_turn>model"""

        # Memory-efficient generation
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Reduced for memory safety
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Clean output
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip()
    
    except Exception as e:
        return f"Gemma-2B Error: {str(e)}"