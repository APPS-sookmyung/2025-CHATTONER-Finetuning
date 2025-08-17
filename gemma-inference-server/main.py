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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gc.collect()
print(" CPU cache cleared")

# PyTorch 내부 캐시도 정리
torch._C._cuda_clearCublasWorkspaces() if torch.cuda.is_available() else None
print(" All caches cleared and ready!")

# .env 파일 로드 시도
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env file loaded")
except ImportError:
    logger.warning("python-dotenv not installed, using system environment variables only")

# HF_TOKEN 확인 및 입력받기
def get_hf_token():
    # 1. config.py에서 확인
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        return config.HF_TOKEN
    
    # 2. 환경변수에서 확인
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        return hf_token
    
    # 3. 직접 입력받기
    print("\n HF_TOKEN이 설정되지 않았습니다!")
    print("Hugging Face 토큰을 입력해주세요:")
    try:
        import getpass
        token = getpass.getpass("HF_TOKEN: ")
        if token.strip():
            return token.strip()
        else:
            raise ValueError("토큰이 입력되지 않았습니다!")
    except KeyboardInterrupt:
        print("\n 토큰 입력이 취소되었습니다.")
        exit(1)
    except Exception as e:
        print(f" 토큰 입력 중 오류: {e}")
        exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma-2B LoRA Inference Server",
    description="Fine-tuned Gemma 2B model inference API",
    version="1.0.0"
)

# 글로벌 변수
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
        # 토큰 확인
        if not config.HF_TOKEN:
            raise ValueError("HF_TOKEN 환경변수가 설정되지 않았습니다!")
        
        logger.info("Logging into Hugging Face...")
        login(config.HF_TOKEN)
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_ID,
            trust_remote_code=True,
            token=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading base model with quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0}, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            config.LORA_MODEL_ID,
            token=True
        )
        
        device = next(model.parameters()).device
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 입력 길이 제한
        if len(request.prompt) > 2000:
            raise HTTPException(status_code=400, detail="Prompt too long (max 2000 characters)")
        
        # 토크나이징
        device = next(model.parameters()).device
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        
        # 생성
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
        
        # 디코딩 (원본 프롬프트 제거)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text[len(request.prompt):].strip()
        
        # 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")
        return GenerateResponse(
            result=result,
            status="success",
            prompt_length=input_length,
            generated_length=len(tokenizer.encode(result))
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

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
    """상태 확인 (health_check와 동일)"""
    return await health_check()

@app.get("/")
async def root():
    return {
        "message": "Gemma-2B LoRA Inference Server is running!",
        "endpoints": ["/generate", "/health"],
        "model": config.LORA_MODEL_ID
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)