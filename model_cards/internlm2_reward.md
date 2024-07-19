# InternLM2-Reward Model Card

## Introduction

**InternLM2-Reward** is a series of reward models trained on the foundation of InternLM2-Chat-SFT. These model have been trained using over 2.4 million preference samples, both human-annotated and AI-synthesized, achieving outstanding performance while ensuring a balance between helpfulness and harmlessness.

### Key Features:

- **Variety of Sizes Available**: Our open-sourced reward models are available in sizes of **1.8B, 7B, and 20B**, each demonstrating exceptional performance across various metrics. We aim for these different-sized models to facilitate research on the scaling laws of reward models, providing valuable insights to the community.
- **Comprehensive Coverage of Preference**: Trained with **2.4 million** preference pairs derived from both human annotations and AI synthesis, covering diverse areas such as dialogue, writing, poetry, summarization, coding, mathematics, etc. It also maintains a balance between helpful and harmless.
- **Multilingual Support**: InternLM2-Reward was trained on high-quality **English and Chinese** preference data, delivering robust performance in both languages.

This model was applied to the RLHF training process of InternLM2-Chat. The reward model training techniques from the [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297) have been open-sourced in XTuner, try it out [here](https://github.com/InternLM/xtuner)!

## Download links

| Model                     | RewardBench Score | Transformers(HF)                                   | ModelScope(HF)                                    | OpenXLab(HF)                                    | Release Date |
| ------------------------- | ----------------- | -------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ------------ |
| **InternLM2-1.8B-Reward** | 80.6              | [🤗internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-1_8b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-1_8b-reward) | 2024-07-19   |
| **InternLM2-7B-Reward**   | 86.6              | [🤗internlm2-7b-reward](https://huggingface.co/internlm/internlm2-7b-reward) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-7b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-7b-reward) | 2024-07-19   |
| **InternLM2-20B-Reward**  | 89.5              | [🤗internlm2-20b-reward](https://huggingface.co/internlm/internlm2-20b-reward) | [<img src="../assets/modelscope_logo.png" width="20px" /> internlm2-20b-reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b-reward/summary) | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-20b-reward) | 2024-07-19   |

## Performance Evaluation on RewardBench

| Models                | Score | Chat | Chat Hard | Safety | Reasoning |
| --------------------- | ----- | ---- | --------- | ------ | --------- |
| InternLM2-20B-Reward  | 89.5  | 98.6 | 74.1      | 89.4   | 95.7      |
| InternLM2-7B-Reward   | 86.6  | 98.6 | 66.7      | 88.3   | 92.8      |
| InternLM2-1.8B-Reward | 80.6  | 95.0 | 58.1      | 81.8   | 87.4      |

- The evaluation is conducted on the [RewardBench](https://github.com/allenai/reward-bench) dataset.
- For a fair comparison, conditional system prompts proposed in our technical report were not included during testing.

## Demo Code

### Basic Usage

We provide some user-friendly APIs for you to use the model. Here is an example of how to use the model to get the reward score of a chat, compare two chats, or rank multiple chats.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "internlm/internlm2-20b-reward",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)

chat_1 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?"}
]
chat_2 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "I have no idea."}
]


# get reward score for a single chat
score1 = model.get_score(tokenizer, chat_1)
score2 = model.get_score(tokenizer, chat_2)
print("score1: ", score1)
print("score2: ", score2)
# >>> score1:  0.767578125
# >>> score2:  -2.22265625


# batch inference, get multiple scores at once
scores = model.get_scores(tokenizer, [chat_1, chat_2])
print("scores: ", scores)
# >>> scores:  [0.767578125, -2.22265625]


# compare whether chat_1 is better than chat_2
compare_res = model.compare(tokenizer, chat_1, chat_2)
print("compare_res: ", compare_res)
# >>> compare_res:  True


# rank multiple chats, it will return the ranking index of each chat
# the chat with the highest score will have ranking index as 0
rank_res = model.rank(tokenizer, [chat_1, chat_2])
print("rank_res: ", rank_res)  # lower index means higher score
# >>> rank_res:  [0, 1]
```

### Best of N Sampling

Here is an example of how to use the reward model to perform best of N sampling.
The code below demonstrates how to select the best response from the candidates generated by the language model.

```python
import torch
from transformers import AutoModel, AutoTokenizer

# prepare the llm model and tokenizer
llm = AutoModel.from_pretrained(
    "internlm/internlm2-chat-7b",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
llm_tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)

# prepare the reward model and tokenizer
reward = AutoModel.from_pretrained(
    "internlm/internlm2-20b-reward",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
reward_tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)

# prepare the chat prompt
prompt = "Write an article about the artificial intelligence revolution."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = llm_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = llm_tokenizer([text], return_tensors="pt").to("cuda")

# generate best of N candidates
num_candidates = 10  # N=10
candidates = []

outputs = llm.generate(
    **model_inputs,
    max_new_tokens=512,
    num_return_sequences=num_candidates,
    pad_token_id=llm_tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)
outputs = outputs[:, model_inputs["input_ids"].shape[1]:]
for i in range(num_candidates):
    candidate = llm_tokenizer.decode(outputs[i], skip_special_tokens=True)
    candidates.append(messages + [{"role": "assistant", "content": candidate}])

rank_indices = reward.rank(reward_tokenizer, candidates)
sorted_candidates = sorted(zip(rank_indices, candidates), key=lambda x: x[0])

## print the ranked candidates
# for i, (rank_index, candidate) in enumerate(sorted_candidates):
#     print(f"------------Rank {i}------------: \n{candidate[-1]['content']}")

# print the best response
best_response = sorted_candidates[0][1][-1]['content']
print(best_response)
```
