# Amazon fashion data 의 description 생성하기

- input 생성 방법 : keyphrase extraction by KeyBERT 

- text generation : GPT-2

## Colab link
https://colab.research.google.com/drive/1f8GGTGI-P5nbgxJYIZhbuAgl5pXI21bE?usp=sharing


## 1. Fine tuning model

model을 fine tuning 하기 위해서는 new line 으로 저장 된 문서가 필요
1) hugging face에서 기본으로 제공되는 wiki text 를 사용하는 방법

```python
!python /content/transformers/examples/pytorch/language-modeling/run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir [output save path] \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1
```

2) 원하는 데이터를 training data로 사용하는 방법

- amazon fashion data의 상품별 description을 newline으로 저장, train, validation, test file 분리

** gpt-2는 fine tuning 없이도 충분히 좋은 결과를 출력함, 생성하려는 데이터에 맞게 모델을 2차적으로 학습시켜 더 좋을 결과를 내기 위해 fine tuning을 진행하는 것

```python
! python /content/transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2 \
    --train_file /data/training data/train.txt \ #train file path
    --validation_file /data/training data/valid.txt \ #validation file path
    --do_train \
    --do_eval \
    --output_dir [output save path] \
    --per_device_eval_batch_size=2 \ #default=8
    --per_device_train_batch_size=2 \ #default=8
    --num_train_epochs 3 \ #default=3
```
## 2. Input text generation

- description data가 존재하지 않은 데이터의 description 생성을 위해 최소한의 input이 필요함

- KeyBERT를 사용하여 input text 생성 : https://github.com/MaartenGr/KeyBERT

- KeyBERT : pretrained BERT 을 사용하여 주어진 문서와 각 어구(n-gram)를 임베딩, 서로의 cosine similarity을 계산해 문서를 대표하는 keyphase를 채택

- amazon fashion data의 상품별 review text들을 합친 것을 input으로 하여 상품별 keyPhrase들을 생성 (description_keyphrases_v1.csv)

![image](https://user-images.githubusercontent.com/53829167/122163900-dea73c80-ceb0-11eb-931c-62f51513fd7b.png)



## 3. text generation

- 1),2) 의 결과를 비교하여 둘 중 더 괜찮은 결과를 사용

1) fine tuning 한 모델로 text 생성

```python
#load input data
import pandas as pd
data = pd.read_csv('/data/input data/description_keyphrases_v1.csv',index_col=0)
```
```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='/[saved fine tuned model path]')
set_seed(42)
result = list()
for i in range(len(data)):
  result.append(generator(data["nlp_keywords"][i], max_length=30, num_return_sequences=1)[0]["generated_text"])
result
```

2) pretrained GPT-2 모델 사용하여 text 생성(finetuning 안한 것)

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = list()
for i in range(len(data)):
  result.append(generator(data["nlp_keywords"][i], max_length=30, num_return_sequences=1)[0]["generated_text"])
result
```
