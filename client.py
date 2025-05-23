"""
이 파일은 추론할 문장은 토큰화하고 임베딩하여 powerformer server로 전송하는 파일이다.
추론할 문장은 무작위의 문장을 뽑아서 추론한다.

클라이언트는 liberate engine을 사용하여 sk, pk, evk를 생성한다.
그리고 추론할 문장을 토큰화하고 임베딩하여 pk, evk와 함께 powerformer server로 전송한다.

powerformer server는 클라이언트로부터 받은 데이터를 복호화하고 추론을 수행한다.
추론 결과는 클라이언트로 반환된다.
"""

# ---------------------------------------------
# 0. 라이브러리 import
# ---------------------------------------------
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import liberate.fhe as fhe

# ---------------------------------------------
# Utility Function for Packing
# ---------------------------------------------
def pack_parallel(matA: np.ndarray, matB: np.ndarray | None = None) -> np.ndarray:
    """
    두 행렬을 짝-홀 슬롯에 교차 삽입하여 (A/B) parallel row-major 벡터 반환.
    B를 생략하면 A를 두 번 넣어 (X/X) 패킹을 만든다.
    """
    if matB is None:
        matB = matA
    v1 = matA.reshape(-1)                # row-major flatten
    v2 = matB.reshape(-1)
    out = np.empty(v1.size + v2.size, dtype=np.complex128)
    out.real[0::2] = v1                  # 짝수 슬롯 ← A
    out.real[1::2] = v2                  # 홀수 슬롯 ← B
    # 허수부는 0
    return out

def prepare_client_data():
    params = fhe.presets.params["gold"].copy()
    # params["devices"] = [0]     
    params["devices"] = ['cpu']          
    engine = fhe.ckks_engine(**params, verbose=True)

    sk  = engine.create_secret_key()
    pk  = engine.create_public_key(sk)
    evk = engine.create_evk(sk)              # 회전·relin·bootstrapping 키 묶음

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    bert_tiny = AutoModel.from_pretrained("prajjwal1/bert-tiny", add_pooling_layer=False)
    bert_tiny.eval()

    sample = load_dataset("glue", "rte", split="validation[:1]")
    sentence1 = sample["sentence1"][0]
    sentence2 = sample["sentence2"][0]
    tokens = tokenizer(sentence1, sentence2, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        embeds = bert_tiny.embeddings(input_ids=tokens["input_ids"], token_type_ids=tokens.get("token_type_ids"))
    X = embeds[0].cpu().numpy()
    vec_parallel = pack_parallel(X)

    pt_in = engine.encode(vec_parallel, padding=True)

    ctx_in = engine.encrypt(pt_in, pk)

    return engine, ctx_in, pk, evk

if __name__ == "__main__":
    engine_instance, encrypted_input, public_key, eval_key = prepare_client_data()
    print("Client.py executed directly. Data prepared.")