import requests
import json
import time

# 런팟 서버 주소
API_URL = "http://localhost:8001"  # 로컬 테스트용
# API_URL = "http://런팟IP:8001"  # 런팟 배포 후

def test_health():
    """서버 상태 확인"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Health check failed: {e}")

def test_generate(prompt: str, max_tokens: int = 100):
    """텍스트 생성 테스트"""
    print(f"Generating text for: '{prompt[:50]}...'")
    
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.7,
        "do_sample": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! (took {end_time - start_time:.2f}s)")
            print(f"Generated: {result['result']}")
            print(f"Prompt length: {result['prompt_length']}")
            print(f"Generated length: {result['generated_length']}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_korean_prompts():
    """한국어 프롬프트 테스트"""
    korean_prompts = [
        "안녕, 오늘 날씨 어때?",
        "한국의 전통 음식에 대해 설명해줘.",
        "AI의 미래에 대해 어떻게 생각해?",
        "파이썬 프로그래밍을 배우고 싶은 초보자에게 조언을 해줘."
    ]
    
    for i, prompt in enumerate(korean_prompts, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}/{len(korean_prompts)}")
        print(f"{'='*50}")
        test_generate(prompt, max_tokens=150)
        time.sleep(2)  # 서버 부하 방지

if __name__ == "__main__":
    print("Gemma Inference Server Test")
    print("="*50)
    
    # 서버 상태 확인
    test_health()
    print()
    
    # 간단한 생성 테스트
    test_generate("Hello, how are you today?", max_tokens=50)
    print()
    
    # 한국어 테스트
    print("🇰🇷 Korean Language Tests")
    test_korean_prompts()
    
    print("\nAll tests completed!")