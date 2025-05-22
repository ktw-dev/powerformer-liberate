"""
distill_powerformer_rte.py
──────────────────────────
교사:  ./final-model/bert-base-finetuned-rte
학생:  BERT-tiny  +  PowerformerEncoder  (이미 구현되어 있다고 가정)
      └── ff_layers: PolyReLU
      └── attn:      BRPmax + HE-friendly matmul

학습 산출물
──────────
 - ./student_powerformer_rte/
   ├─ pytorch_model.bin
   ├─ powerformer_encoder_config.json
   └─ tokenizer/
"""

# ------------------------------------------------------------------
# 0.  라이브러리 & 헬퍼
# ------------------------------------------------------------------
import os, math, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoConfig, AutoModel,
                          AutoModelForSequenceClassification,
                          get_scheduler)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# reproducibility (선택)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------------
# 1.  교사(Teacher) 로드
# ------------------------------------------------------------------
teacher_ckpt = "./bert-base-finetuned-rte/checkpoint-624"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_ckpt)
teacher_model.to(device).eval()

num_labels = teacher_model.config.num_labels  # 2 (entail / not-entail)

# ------------------------------------------------------------------
# 2.  학생(Student) – Powerformer BERT-tiny
#      (PowerformerEncoder 클래스를 사용한다고 가정)
# ------------------------------------------------------------------
from powerformer_encoder import PowerformerEncoder  # ← 이미 작성된 모듈

student_cfg = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
# 인코더 교체
bert_tiny_body          = AutoModel.from_pretrained("prajjwal1/bert-tiny")
# Get the class of the original encoder layer
original_encoder_layer_cls = type(bert_tiny_body.encoder.layer[0])
powerformer_encoder_cls = original_encoder_layer_cls # Use original BERT layer

# encoder.layer  전체를 PowerformerEncoder 로 교체
power_layers = nn.ModuleList([
    powerformer_encoder_cls(student_cfg) # Standard BertLayer usually takes full config
    # Or, if it takes individual args, verify them:
    # powerformer_encoder_cls(hidden_size=student_cfg.hidden_size, ...)
    for _ in range(student_cfg.num_hidden_layers)
])
bert_tiny_body.encoder.layer = power_layers

# 학생 분류 헤드
class StudentPowerformerForRTE(nn.Module):
    def __init__(self, bert_body, num_labels, dropout_p=0.1):
        super().__init__()
        self.bert = bert_body
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(bert_body.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None,
                token_type_ids=None, labels=None, output_hidden=False):
        out = self.bert(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        pooled = out.pooler_output if out.pooler_output is not None \
                 else out.last_hidden_state[:, 0]          # [CLS]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        if output_hidden:
            return (loss, logits, pooled) if loss is not None else (logits, pooled)
        return (loss, logits) if loss is not None else (logits,)

student_model = StudentPowerformerForRTE(
    bert_body=bert_tiny_body,
    num_labels=num_labels,
    dropout_p=teacher_model.config.classifier_dropout or 0.1
).to(device)

# ------------------------------------------------------------------
# 3.  데이터셋 (GLUE-RTE)
# ------------------------------------------------------------------
raw = load_dataset("glue", "rte")

def tok_fn(ex):
    return teacher_tokenizer(ex["sentence1"], ex["sentence2"],
                             truncation=True, padding="max_length", max_length=128)

tok_ds = raw.map(tok_fn, batched=True)
tok_ds = tok_ds.remove_columns(["sentence1", "sentence2", "idx"])
tok_ds = tok_ds.rename_column("label", "labels")
tok_ds.set_format("torch")

train_loader = DataLoader(tok_ds["train"], shuffle=True, batch_size=16)
val_loader   = DataLoader(tok_ds["validation"], batch_size=32)

# ------------------------------------------------------------------
# 4.  증류 하이퍼파라미터
# ------------------------------------------------------------------
T            = 3.0   # temperature
alpha_kd     = 0.5   # KD loss weight
learning_rate= 1e-5
epochs       = 15 

optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=1e-2)
num_steps = epochs * len(train_loader)
lr_sched  = get_scheduler("linear", optimizer=optimizer,
                          num_warmup_steps=0, num_training_steps=num_steps)

kd_loss_f = nn.KLDivLoss(reduction="batchmean")

# ------------------------------------------------------------------
# 5.  학습 루프
# ------------------------------------------------------------------
for epoch in range(epochs):
    student_model.train()
    tot_loss = tot_kd = tot_ce = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Teacher logits
        with torch.no_grad():
            t_logits = teacher_model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"],
                                     token_type_ids=batch.get("token_type_ids")).logits

        # Student forward
        s_loss_ce, s_logits = student_model(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            token_type_ids=batch.get("token_type_ids"),
                                            labels=batch["labels"])
        # KD loss
        kd_loss = kd_loss_f(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1)
        ) * (T ** 2)

        loss = alpha_kd * kd_loss + (1 - alpha_kd) * s_loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()

        tot_loss += loss.item();  tot_kd += kd_loss.item();  tot_ce += s_loss_ce.item()

    print(f"[Epoch {epoch+1}]  Loss={tot_loss/len(train_loader):.4f}  "
          f"KD={tot_kd/len(train_loader):.4f}  CE={tot_ce/len(train_loader):.4f}")

    # 간단 검증
    student_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = student_model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   token_type_ids=batch.get("token_type_ids"))[0]
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total   += preds.size(0)
    print(f"  → Val Acc: {correct/total:.4%}")

# ------------------------------------------------------------------
# 6.  가중치 저장 (Inference/FHE 용)
# ------------------------------------------------------------------
save_dir = "./student_powerformer_rte_2"
os.makedirs(save_dir, exist_ok=True)

# 학생 모델 전체 저장 (PyTorch 표준 방식)
torch.save(student_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

# PowerformerEncoder 설정 저장
powerformer_config = {
    "hidden_size": student_cfg.hidden_size,
    "num_attention_heads": student_cfg.num_attention_heads,
    "intermediate_size": student_cfg.intermediate_size,
    "num_hidden_layers": student_cfg.num_hidden_layers,
    # 아래 값들은 powerformer_encoder.py의 __init__에서 params로 받는 값들과 일치해야 합니다.
    # 예시 값이며, 실제 PowerformerEncoder 구현에 맞게 조정 필요
    "brpmax": {"p": 2, "c": 2.0, "inv_Rd": 0.03125},
    "relu_poly": {"a1": 0.5, "a3": 0.125},
    # LayerNorm 상수는 PowerformerEncoder가 내부적으로 사용하거나,
    # FHE 엔진 로딩 시 주입될 수 있습니다. 여기서는 예시로 export_cfg에서 가져옴.
    # "layernorm_eps": 1e-5 # 예시. PowerformerEncoder 설계에 따라 다름
}
json.dump(powerformer_config, open(os.path.join(save_dir, "powerformer_encoder_config.json"), "w"), indent=2)


# 인코더 레이어의 가중치를 load_layer_weights가 기대하는 .npy 형태로 저장
# student_model.bert.encoder.layer 가 power_layers 임.
# 각 layer가 PowerformerEncoder의 __init__에 전달될 수 있는 가중치를 포함하는 nn.Module이라고 가정.
# 실제로는 PowerformerEncoder가 직접 nn.Parameter를 가질 수도 있고,
# 혹은 PowerformerEncoder를 감싸는 nn.Module (예: PowerformerBertLayer)이 nn.Parameter를 가질 수 있음.
# 여기서는 후자를 가정하고, 해당 레이어에서 가중치를 추출한다고 가정.

# Prajjwal1/bert-tiny의 구조를 기반으로, 각 PowerformerBertLayer가 다음을 가질 것으로 가정:
# - self.attention.self.query (nn.Linear) -> WQ
# - self.attention.self.key (nn.Linear) -> WK
# - self.attention.self.value (nn.Linear) -> WV
# - self.attention.output.dense (nn.Linear) -> WO
# - self.attention.output.LayerNorm (nn.LayerNorm) -> ln_attn
# - self.intermediate.dense (nn.Linear) -> W1 (FFN의 첫번째)
# - self.output.dense (nn.Linear) -> W2 (FFN의 두번째)
# - self.output.LayerNorm (nn.LayerNorm) -> ln_ffn

for i, layer_module in enumerate(student_model.bert.encoder.layer):
    layer_save_dir = os.path.join(save_dir, f"layer{i:02d}")
    os.makedirs(layer_save_dir, exist_ok=True)

    # WQ, WK, WV: (hidden_size, hidden_size) -> 각 헤드 분리 전
    # load_layer_weights는 (num_heads * head_dim, hidden_size)로 받고 split함.
    # bert-tiny: hidden_size=128, num_heads=2, head_dim=64.
    # nn.Linear.weight는 (out_features, in_features) 임.
    # Q,K,V의 Linear layer는 (hidden_size, hidden_size) -> (128, 128)
    wq_weight = layer_module.attention.self.query.weight.detach().cpu().numpy()
    wk_weight = layer_module.attention.self.key.weight.detach().cpu().numpy()
    wv_weight = layer_module.attention.self.value.weight.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "WQ.npy"), wq_weight)
    np.save(os.path.join(layer_save_dir, "WK.npy"), wk_weight)
    np.save(os.path.join(layer_save_dir, "WV.npy"), wv_weight)

    # WO: attention output dense layer. (hidden_size, hidden_size)
    # load_layer_weights는 (hidden_size, num_heads * head_dim)으로 받고 split함 (열 기준).
    wo_weight = layer_module.attention.output.dense.weight.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "WO.npy"), wo_weight)

    # W1 (FFN intermediate): (intermediate_size, hidden_size)
    w1_weight = layer_module.intermediate.dense.weight.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "W1.npy"), w1_weight)

    # W2 (FFN output): (hidden_size, intermediate_size)
    w2_weight = layer_module.output.dense.weight.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "W2.npy"), w2_weight)

    # LayerNorm attention: (gamma, beta)
    ln_attn_gamma = layer_module.attention.output.LayerNorm.weight.detach().cpu().numpy()
    ln_attn_beta = layer_module.attention.output.LayerNorm.bias.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "ln_attn.npy"), np.stack([ln_attn_gamma, ln_attn_beta]))

    # LayerNorm FFN: (gamma, beta)
    ln_ffn_gamma = layer_module.output.LayerNorm.weight.detach().cpu().numpy()
    ln_ffn_beta = layer_module.output.LayerNorm.bias.detach().cpu().numpy()
    np.save(os.path.join(layer_save_dir, "ln_ffn.npy"), np.stack([ln_ffn_gamma, ln_ffn_beta]))

    # PowerformerEncoder __init__에서 w로 받는 가중치들은 이미 FHE 인코딩된 Plaintext임.
    # 여기서 저장하는 .npy 파일들은 FHE 인코딩 전의 PyTorch 텐서들임.
    # load_layer_weights 함수는 이 .npy들을 로드하여 FHE 인코딩을 수행함.
    # 따라서 위의 저장은 load_layer_weights의 입력 형식에 맞춘 것임.

# 토크나이저(동일 vocab) 저장: 클라이언트 임베딩 재현용
teacher_tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

print(f"\n✓ Distillation complete. Student weights and Powerformer config saved to {save_dir}")