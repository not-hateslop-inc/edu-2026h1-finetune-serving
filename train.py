import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

LABELS = ["긍정", "불만", "요청"]
name = "klue/roberta-small"
df = pd.read_csv("data/feedback_labeled.csv")
df["labels"] = df.label.map(LABELS.index)
tok = AutoTokenizer.from_pretrained(name)

def to_ds(split):
    ds = Dataset.from_pandas(df[df.split == split][["text", "labels"]].reset_index(drop=True))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    name, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
args = TrainingArguments("out/feedback", num_train_epochs=8, per_device_train_batch_size=8,
                         learning_rate=5e-5, eval_strategy="epoch", report_to="none")
t = Trainer(model=model, args=args, train_dataset=to_ds("train"), eval_dataset=to_ds("test"), processing_class=tok)
t.train()
t.save_model("out/feedback/model")
tok.save_pretrained("out/feedback/model")
