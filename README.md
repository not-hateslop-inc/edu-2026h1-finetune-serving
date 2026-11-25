# edu-2026h1-finetune-serving

2026 상반기 15회차 **LLM Fine-tuning과 Model Serving** (EDU-2026-H1-15, 2026-06-11) 실습 저장소입니다.

| 파일 | 내용 |
| --- | --- |
| `train.py` | 교육 피드백 분류기 학습 (HF Trainer) |
| `data/feedback_labeled.csv` | 라벨 있는 피드백 40건 (가상) |
| `serve.py` | FastAPI로 `/classify` 제공 |
| `client.py` | API 호출 예시 |
