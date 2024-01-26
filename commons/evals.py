import torch
from torch.utils.data import DataLoader
from commons.models import ModelZoo, get_cached_model, get_cached_tokenizer
from datasets import load_dataset
from torch.nn import functional as F

dataset = load_dataset("Anthropic/hh-rlhf")
dataset = dataset.with_format("torch")
batch_size = 16
split = "train"
dataloader = DataLoader(dataset[split], batch_size=batch_size)

model_name = ModelZoo.DEBERTA_V3_LARGE_V2
model = get_cached_model(model_name)
tokenizer = get_cached_tokenizer(model_name)
total_chosen = 0
for idx, batch in enumerate(dataloader):
    # TODO check if need to split the raw string into dialogues
    chosen = tokenizer(
        batch["chosen"], return_tensors="pt", padding=True, truncation=True
    )
    rejected = tokenizer(
        batch["rejected"], return_tensors="pt", padding=True, truncation=True
    )

    chosen_logits = model(**chosen).logits
    rejected_logits = model(**rejected).logits
    # cat over batch dim, for 'chosen' and 'rejected'
    logits = torch.cat([chosen_logits, rejected_logits], dim=1)
    # check if chosen is more likely than rejected
    probs = F.softmax(logits, dim=-1)

    values, indices = torch.max(probs, dim=-1)
    num_chosen = sum(indices == 0)
    total_chosen += num_chosen

accuracy = total_chosen / len(dataset[split])
print(f"Accuracy: {accuracy}")
