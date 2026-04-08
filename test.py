from datasets import load_dataset
dataset = load_dataset(
    "allenai/wildjailbreak",
    "train",
    streaming=True
)
print(dataset)