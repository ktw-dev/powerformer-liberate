import json
import os
from powerformer_encoder import PowerformerEncoder, load_layer_weights

# Configuration
MODEL_DIR = "./student_powerformer_rte"  # 학습된 모델 가중치가 저장된 디렉토리
CONFIG_FILE = "powerformer_encoder_config.json"  # 모델 설정 파일명

def handle_client_data(engine_instance, encrypted_input, public_key, eval_key):
    """
    클라이언트로부터 받은 암호화된 데이터를 처리하여 FHE 추론을 수행하는 메인 함수
    
    Args:
        engine_instance: CKKS 암호화 엔진 인스턴스
        encrypted_input: 암호화된 입력 데이터 (ciphertext)
        public_key: 공개키
        eval_key: 평가키 (회전, 재선형화 등을 위한 키 번들)
    
    Returns:
        추론 결과가 담긴 암호문 또는 None (실패 시)
    """
    print(f"Received data. Starting FHE inference with model from {MODEL_DIR}")

    # 모델 설정 파일 로드
    config_path = os.path.join(MODEL_DIR, CONFIG_FILE)
    if not os.path.exists(config_path):
        print(f"ERROR: Powerformer config file not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        powerformer_config = json.load(f)

    # 숨겨진 레이어 수 확인
    num_hidden_layers = powerformer_config.get("num_hidden_layers")
    if num_hidden_layers is None:
        print(f"ERROR: 'num_hidden_layers' not found in powerformer_encoder_config.json")
        return None

    # 각 인코더 레이어 초기화
    encoder_layers = []
    # BRPmax와 ReLU 다항식 매개변수 준비
    params_for_encoder = {
        "brpmax": powerformer_config["brpmax"],
        "relu_poly": powerformer_config["relu_poly"]
    }

    # 각 레이어별로 가중치 로드 및 인코더 생성
    for i in range(num_hidden_layers):
        layer_dir = os.path.join(MODEL_DIR, f"layer{i:02d}")
        if not os.path.isdir(layer_dir):
            print(f"ERROR: Layer weights directory not found: {layer_dir}")
            return None
        
        # 레이어 가중치 로드 (WQ, WK, WV, WO, W1, W2, LN1, LN2)
        layer_weights = load_layer_weights(layer_dir, engine_instance)
        
        # PowerformerEncoder 인스턴스 생성
        encoder = PowerformerEncoder(
            engine=engine_instance,
            evk=eval_key,
            params=params_for_encoder, 
            w=layer_weights
        )
        encoder_layers.append(encoder)

    # 순차적으로 각 레이어를 통과하며 FHE 추론 수행
    current_ct = encrypted_input  # 초기 입력은 클라이언트로부터 받은 암호화된 데이터
    for idx, layer in enumerate(encoder_layers):
        print(f"Processing layer {idx+1}/{num_hidden_layers}")
        current_ct = layer(current_ct)  # 각 레이어의 __call__ 메서드 호출

    print(f"FHE Inference Complete.")
    return current_ct  # 최종 추론 결과 반환

if __name__ == '__main__':
    # 독립 실행 방지 - 이 스크립트는 클라이언트가 FHE 객체를 제공할 때만 동작
    print(f"server.py is not meant to be run directly without a client providing FHE objects.")
