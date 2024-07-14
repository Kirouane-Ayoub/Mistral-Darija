import argparse
import time

import settings
import torch
from dataset import TextDataset, get_new_tokenizer
from model import Mistral, ModelArgs
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

model_args = ModelArgs(
    dim=4096,  # Embedding dimension
    n_layers=4,  # Number of Transformer layers
    head_dim=128,  # Head dimension for multi-head attention
    hidden_dim=14336,  # Dimension of hidden layer in the feedforward network
    n_heads=32,  # Number of attention heads
    n_kv_heads=32,  # Number of key/value heads (can be different from n_heads)
    norm_eps=1e-5,  # Epsilon value for normalization
    vocab_size=32000,  # Size of your vocabulary
    rope_theta=10000,  # Base value for Rotary Position Embeddings
)

# Create the Mistral model
model = Mistral(model_args)

# Load the tokenizer
new_tokenizer = get_new_tokenizer()


# Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)


def train(
    model: Mistral,
    train_data: DataLoader,
    val_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clip_grad_norm: float = 1.0,
    lr_scheduler=None,
):
    """Trains the Mistral model.

    Args:
        model: The Mistral model to train.
        train_data: A DataLoader for the training dataset.
        optimizer: The optimizer to use for training.
        epochs: The number of training epochs.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        clip_grad_norm: The maximum norm of the gradients to clip.
        lr_scheduler: An optional learning rate scheduler.
    """

    model = model.to(device)
    model.train()

    print("START Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        start_time = time.time()

        for batch in tqdm(train_data, leave=False):
            input_ids, labels = batch

            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(input_ids)

            # Calculate loss (use cross-entropy loss for language modeling)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, model.vocab_size), labels.view(-1))

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Update weights
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step(loss.detach().item())

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        elapsed_time = time.time() - start_time
        print(f"Average loss: {avg_loss:.4f} | Elapsed time: {elapsed_time:.2f}s")

        # Evaluation Phase
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(val_data):
                # Get input_ids and labels from the batch
                input_ids, labels = batch
                input_ids = input_ids.to(device)  # Send input_ids to the device
                labels = labels.to(device)  # Send labels to the device

                # Forward pass
                outputs, _ = model(input_ids)

                # Calculate loss
                loss = F.cross_entropy(
                    outputs.view(-1, model.vocab_size),
                    labels.view(-1),
                    ignore_index=new_tokenizer.pad_token_id,
                )
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(val_data)
        print(f"Epoch: {epoch+1}, Evaluation Loss: {avg_eval_loss:.4f}")
    torch.save(model.state_dict(), settings.MODEL_SAVE_PATH)
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=settings.TRAIN_DATA_PATH,
        required=True,
        help="Path to the training data.",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=settings.EVAL_DATA_PATH,
        required=True,
        help="Path to the evaluation data.",
    )

    args = parser.parse_args()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create train dataset and dataloaders
    train_dataset = TextDataset(args.train_data, new_tokenizer, max_length=512)
    train_loader = DataLoader(
        train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True
    )

    # Create dataset and dataloaders
    eval_dataset = TextDataset(args.eval_data, new_tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        clip_grad_norm=1.0,
        lr_scheduler=lr_scheduler,
    )
