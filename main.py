"""
이 파일은 client.py와 server.py를 실행하는 파일이다.

client.py는 클라이언트를 실행하는 파일이다.
server.py는 서버를 실행하는 파일이다.

client.py를 실행하면 추론을 위해 임베딩된 문장과 pk, evk를 서버로 전송한다.
server.py는 클라이언트로부터 받은 데이터를 복호화하고 추론을 수행한다.
추론 결과는 클라이언트로 반환된다.
"""

import client
import server
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # CUDA 연산을 동기식으로 만들어 디버깅 용이
os.environ['TORCH_USE_CUDA_DSA'] = "1"    # 디바이스 측 어설션 활성화

import torch

if __name__ == "__main__":
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    print(f"--- {__file__} --- PyTorch version: {torch.__version__}")
    print(f"--- {__file__} --- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"--- {__file__} --- CUDA version: {torch.version.cuda}")
        print(f"--- {__file__} --- GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"--- {__file__} --- Initial GPU Memory Summary ---")
        print(torch.cuda.memory_summary(device=0))

    prepared_data = client.prepare_client_data()
    if prepared_data is None:
        print(f"Client data preparation failed. Exiting.")
        exit()
        
    engine_instance, encrypted_input, public_key, eval_key = prepared_data

    if hasattr(server, "handle_client_data"):
        server_result_ct = server.handle_client_data(
            engine_instance=engine_instance,
            encrypted_input=encrypted_input,
            public_key=public_key,
            eval_key=eval_key
        )

        
        if server_result_ct is not None:
            print(f"Server processing complete. Result ciphertext type: {type(server_result_ct)}")
        else:
            print(f"Server processing failed")
    else:
        print(f"Server function 'handle_client_data' not found in server.py. Skipping server processing.")
