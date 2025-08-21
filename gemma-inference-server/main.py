from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import config
import logging
import os
import gc
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

app = FastAPI(
    title="Gemma-2B LoRA Inference Server",
    description="Fine-tuned Gemma 2B model inference API",
    version="1.0.0"
)

tokenizer = None
model = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS
    temperature: float = config.DEFAULT_TEMPERATURE
    do_sample: bool = config.DEFAULT_DO_SAMPLE

class GenerateResponse(BaseModel):
    result: str
    status: str
    prompt_length: int
    generated_length: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not set")
        
        login(hf_token)
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_ID,
            trust_remote_code=True,
            token=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            config.LORA_MODEL_ID,
            token=True,
            device_map="auto",
        )
        
        device = next(model.parameters()).device
        logger.info(f"Model loaded on {device}")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise e

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if len(request.prompt) > 2000:
            raise HTTPException(status_code=400, detail="Prompt too long")
        
        device = next(model.parameters()).device
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text[len(request.prompt):].strip()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return GenerateResponse(
            result=result,
            status="success",
            prompt_length=input_length,
            generated_length=len(tokenizer.encode(result))
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    device = "none"
    if model is not None:
        device = str(next(model.parameters()).device)
    
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_id=config.LORA_MODEL_ID,
        device=device
    )

@app.get("/status", response_model=HealthResponse)
async def status():
    return await health_check()

@app.get("/")
async def root():
    return {
        "message": "Gemma-2B LoRA Inference Server",
        "endpoints": ["/generate", "/health"],
        "model": config.LORA_MODEL_ID
    }

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)