import pandas as pd
from datasets import Dataset , DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer ,AutoModel ,Trainer,AutoConfig , AutoModelForSequenceClassification ,TrainingArguments
import torch
import evaluate
import numpy as np

model_checkpoint = 'bert-base-uncased'
import pandas as pd
data = pd.read_csv('data.csv')
data =data.drop('Unnamed: 0' ,axis=1)
lable2id = {value:index for index, value in enumerate(data['status'].unique())}
data['label'] = data['status'].map(lable2id)
data['statement'] = data['statement'].fillna('')

# id2label = {val:key for key , val in lable2id.items()}
data['word_per_statement'] =  data['statement'].apply(lambda x: len(str(x).split()))
data.head(100)
train , test = train_test_split(data , test_size=.3 , random_state=42 ,stratify=data['label'])
test , validate = train_test_split(test , test_size=.5 , random_state=42 ,stratify=test['label'])

dataset = DatasetDict({
   'train': Dataset.from_pandas(train,preserve_index=False),
  'test': Dataset.from_pandas(test , preserve_index=False),
  'validate': Dataset.from_pandas(validate, preserve_index=False)
})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def tokenize(batch):
  temp = tokenizer(batch['statement'], padding=True, truncation=True)
  return temp
sentiment_encoder = dataset.map(tokenize, batched=True, batch_size=None)
sentiment_encoder['train'][0]
label2id = {i['status']:str(i['label']) for i in sentiment_encoder['train']}
id2label = {val:key for key , val in label2id.items()}
num_labeld = len(label2id)
model = AutoModel.from_pretrained(model_checkpoint)
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labeld, id2label=id2label, label2id=label2id)
model =AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
model.to(device)

batch_size = 64
training_dir = 'bert-base-sa-model-train'

train_args = TrainingArguments(
    output_dir=training_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy='epoch',
    disable_tqdm=False
)

accuracy =evaluate.load('accuracy')
def compute_matrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(model=model , args=train_args,compute_metrics=compute_matrics,
                  train_dataset=sentiment_encoder['train'],
                  eval_dataset=sentiment_encoder['validate'],
                  tokenizer=tokenizer)

trainer.train()

trainer.save_model('bert-base-sa-mental-uncased')
preds_output = trainer.predict(sentiment_encoder['test'])

tokenizer.save_pretrained("bert-base-sa-mental-uncased")
