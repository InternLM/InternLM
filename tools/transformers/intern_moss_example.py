import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from moss_002_sft import get_dataset, collate_fn

model_path = "model_path"
data_dir = "moss_002_sft"
data_num = -1
test_size = 10
train_batch_size = 1
epochs = 5
val_per_steps = 1000
lr = 9e-6
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=32, lora_dropout=0.1,
    target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
)


# model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = get_peft_model(model, peft_config)
model.cuda()

# dataset
train_dataset, val_dataset = get_dataset(tokenizer, data_dir, num=data_num, test_size=test_size)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

optimizer = torch.optim.AdamW(model.parameters(), lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 1000, epochs * len(train_dataloader)
)

# train
fp = open("output", "w")
model.train()
for epoch in tqdm(range(epochs), desc="Traning Epoch"):
    batch_bar = tqdm(train_dataloader, desc="Training Batch")
    for step, batch in enumerate(batch_bar):
        batch = {k:v.cuda() for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)

        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_bar.set_postfix({"loss": loss.item()})
        if (step + 1) % val_per_steps == 0:
            fp.write(f"Epoch {epoch} Batch {step}: Loss={loss.item()}\n")
            for i in tqdm(range(len(val_dataset)), desc="Generating"):
                data, label = val_dataset[i]
                prefix = tokenizer.decode(data.tolist(), skip_special_tokens=True)
                try:
                    generate = model.generate(input_ids=data.unsqueeze(0).cuda(), temperature=0.7, top_k=50, do_sample=True, repetition_penalty=1.02, max_new_tokens=100, top_p=0.9)
                    text = tokenizer.decode(generate[0].tolist(), skip_special_tokens=True)
                    text = text.replace(prefix, "")
                    fp.write(f"Prefix: {prefix}\nGenerated: {text}" + "\n---------------------------------\n")
                except Exception as e:
                    fp.write(f"Prefix: {prefix}\nError: {e}" + "\n---------------------------------\n")
            fp.write("\n==============================\n")
            model.train()
            torch.cuda.empty_cache()
