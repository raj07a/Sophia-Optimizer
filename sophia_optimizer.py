from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import time
from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# Dataset Preparation
# -----------------------------
def load_and_tokenize_dataset():
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )

    # Tokenize the datasets
    tokenized_train = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_train, tokenized_val, tokenizer

# Load tokenized datasets and tokenizer
train_dataset, val_dataset, tokenizer = load_and_tokenize_dataset()

# Data collator for handling batches
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Use a smaller subset for training and validation (optional)
train_dataset = train_dataset.select(range(100))  # 100 samples for training
val_dataset = val_dataset.select(range(20))  # 20 samples for validation

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

# -----------------------------
# Optimizer Implementations
# -----------------------------
class SophiaOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-12, weight_decay=0.0, gamma=0.01, clip_threshold=1.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.clip_threshold = clip_threshold
        self.m = None
        self.h = None
        self.t = 0

    def estimate_diagonal_hessian(self, grads):
        v = np.random.choice([-1, 1], size=grads.shape)
        hessian_diag = v * grads * v
        return np.abs(hessian_diag)

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.h = np.ones_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        if self.t % 10 == 0:  # Dynamic Hessian updates every 10 steps
            hessian_diag = self.estimate_diagonal_hessian(grads)
            self.h = self.beta2 * self.h + (1 - self.beta2) * hessian_diag

        preconditioned_update = self.m / (self.h * self.gamma + self.epsilon)
        clipped_update = np.clip(preconditioned_update, -self.clip_threshold, self.clip_threshold)

        params -= self.learning_rate * clipped_update

        if self.weight_decay > 0:
            params -= self.weight_decay * params

        return params

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, optimizer, scaler, n_epochs, train_dataloader, val_dataloader, device):
    results = {"train_loss": [], "val_loss": [], "time": [], "val_accuracy": []}
    detailed_results = []

    for epoch in range(n_epochs):
        start_time = time.time()

        # Training loop
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            inputs = {key: val.to(device) for key, val in batch.items()}

            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == inputs["input_ids"]).sum().item()
                    total += inputs["input_ids"].numel()

        val_loss /= len(val_dataloader)
        val_accuracy = correct / total

        end_time = time.time()

        # Log results
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_accuracy)
        results["time"].append(end_time - start_time)

        # Store detailed results
        detailed_results.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Time (s)": end_time - start_time
        })

    return results, detailed_results

# -----------------------------
# Comparison Function
# -----------------------------
def compare_optimizers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Optimizers
    optimizers = {
        "SGD": torch.optim.SGD(model.parameters(), lr=0.01),
        "Adam": torch.optim.Adam(model.parameters(), lr=0.001),
        "Sophia": torch.optim.AdamW(model.parameters(), lr=0.001)  # Replace with Sophia implementation if available
    }

    scaler = GradScaler()  # For mixed precision training

    n_epochs = 3
    all_results = {}
    detailed_results = []

    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name} optimizer...")
        results, detailed = train_model(model, optimizer, scaler, n_epochs, train_dataloader, val_dataloader, device)
        all_results[name] = results
        for entry in detailed:
            entry["Optimizer"] = name
            detailed_results.append(entry)

    # Display detailed results
    print("\nDetailed Results per Epoch:")
    detailed_table = PrettyTable()
    detailed_table.field_names = ["Optimizer", "Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Time (s)"]
    for result in detailed_results:
        detailed_table.add_row([result["Optimizer"], result["Epoch"], f"{result['Train Loss']:.4f}", f"{result['Validation Loss']:.4f}", f"{result['Validation Accuracy']:.4f}", f"{result['Time (s)']:.2f}"])
    print(detailed_table)

    # Final accuracy comparison
    print("\nFinal Accuracy Comparison:")
    accuracy_table = PrettyTable()
    accuracy_table.field_names = ["Optimizer", "Final Validation Accuracy"]
    for name in optimizers.keys():
        accuracy_table.add_row([name, f"{all_results[name]['val_accuracy'][-1]:.4f}"])
    print(accuracy_table)

    # Visualizations
    plt.figure(figsize=(12, 6))
    for name in optimizers.keys():
        plt.plot(all_results[name]["train_loss"], label=f"{name} Train Loss")
        plt.plot(all_results[name]["val_loss"], label=f"{name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer Loss Convergence on WikiText-103")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    for name in optimizers.keys():
        plt.plot(all_results[name]["val_accuracy"], label=f"{name} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    avg_times = [np.mean(all_results[name]["time"]) for name in optimizers.keys()]
    plt.figure(figsize=(10, 6))
    plt.bar(optimizers.keys(), avg_times, color=['blue', 'orange', 'green'])
    plt.ylabel("Average Time (s)")
    plt.title("Average Time Per Epoch by Optimizer")
    plt.show()

# Run the comparison
compare_optimizers()
