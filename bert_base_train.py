from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate

# GLUE-RTE 데이터셋 로드
dataset = load_dataset("glue", "rte")

# BERT base 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 데이터 전처리 함수 정의
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

# 데이터셋 전처리
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 데이터 콜레이터 정의 (동적 패딩 사용)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 평가 메트릭 정의
metric = evaluate.load("glue", "rte")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# BERT base 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", 
    num_labels=2
)

# 학습 인자 설정 - Early Stopping을 위한 설정 추가
training_args = TrainingArguments(
    output_dir="./results/bert-base-finetuned-rte",
    learning_rate=3e-5,  # 요청하신 learning rate
    per_device_train_batch_size=32,  # 요청하신 batch size
    per_device_eval_batch_size=32,
    num_train_epochs=50,  # 10 에포크로 변경
    weight_decay=1e-2,  # 요청하신 weight decay
    eval_strategy="epoch",  # Early Stopping을 위해 'epoch'에서 'steps'로 변경
    # eval_steps=10,  # 15 스텝마다 평가
    save_strategy="epoch",  # 저장 전략도 steps로 변경
    # save_steps=150,  # 50 스텝마다 저장
    load_best_model_at_end=True,  # Early Stopping을 위해 필요
    metric_for_best_model="accuracy",  # 성능 지표로 accuracy 사용
    push_to_hub=False,
)

# Trainer 초기화 - EarlyStoppingCallback 추가
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 3번의 평가동안 개선이 없으면 중단
)

# 모델 파인튜닝
trainer.train()

# 모델 평가
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 모델 저장
trainer.save_model("./final-model/bert-base-finetuned-rte")
