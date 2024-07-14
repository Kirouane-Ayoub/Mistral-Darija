import settings
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# Function to get training data
def get_training_corpus():
    dataset = load_dataset("text", data_files={"train": settings.TRAIN_DATA_PATH})
    for i in range(0, len(dataset["train"]), 1000):
        yield dataset["train"][i : i + 1000]["text"]


def get_new_tokenizer(tokinizer_name=settings.TOKENIZER_NAME):
    # Load the base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(tokinizer_name)

    # Train the new tokenizer
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        get_training_corpus(), vocab_size=1000
    )  # Change this to the size of your vocabulary

    # Save the new tokenizer
    new_tokenizer.save_pretrained(
        "darija_tokenizer"
    )  # Change this to the name of your new tokenizer
    return new_tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, encoding="utf-8") as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )  # Ensure PyTorch Tensor output

        input_ids = encoding["input_ids"].squeeze()

        # Assuming you want to use the input_ids as labels for language modeling
        # create a copy of the input_ids for the labels
        labels = input_ids.clone()

        labels[:-1] = input_ids[1:]  # Shift labels
        return input_ids, labels  # Return both input_ids and labels
