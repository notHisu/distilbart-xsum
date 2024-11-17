import torch
from datasets import load_dataset
import pandas as pd

# Load XSum dataset
print("Loading XSum dataset...")
dataset = load_dataset("xsum")
print("Dataset loaded successfully!")

dataset = dataset.remove_columns(["id"])

# Define strata for stratified sampling
bins = [0, 500, 1500, float("inf")]
labels = ["short", "medium", "long"]

# Stratified sampling function
def stratified_sample(strata, num_samples_per_stratum):
    unique_labels = torch.unique(strata)
    indices = []
    for label in unique_labels:
        label_indices = torch.nonzero(strata == label).flatten()
        print(f"Label: {label}, Number of documents: {len(label_indices)}")
        if len(label_indices) < num_samples_per_stratum:
            sampled_indices = label_indices
        else:
            sampled_indices = label_indices[torch.randperm(len(label_indices))[:num_samples_per_stratum]]
        indices.extend(sampled_indices.tolist())
    return indices

# Sample the dataset to get a subset
print("Sampling the dataset...")
train_strata = torch.bucketize(torch.tensor([len(doc) for doc in dataset['train']['document']]), torch.tensor(bins), right=True)
val_strata = torch.bucketize(torch.tensor([len(doc) for doc in dataset['validation']['document']]), torch.tensor(bins), right=True)
test_strata = torch.bucketize(torch.tensor([len(doc) for doc in dataset['test']['document']]), torch.tensor(bins), right=True)

sampled_train_indices = stratified_sample(train_strata, min(100 // len(labels), len(dataset['train'])))
sampled_val_indices = stratified_sample(val_strata, min(10 // len(labels), len(dataset['validation'])))
sampled_test_indices = stratified_sample(test_strata, min(10 // len(labels), len(dataset['test'])))

sampled_train_dataset = dataset['train'].select(sampled_train_indices)
sampled_val_dataset = dataset['validation'].select(sampled_val_indices)
sampled_test_dataset = dataset['test'].select(sampled_test_indices)
print("Dataset sampled successfully!")

train_dataset_name = "sampled_train_dataset.csv"
val_dataset_name = "sampled_val_dataset.csv"
test_dataset_name = "sampled_test_dataset.csv"

# Convert the datasets to DataFrames
sampled_train_dataset_df = pd.DataFrame(sampled_train_dataset)
sampled_val_dataset_df = pd.DataFrame(sampled_val_dataset)
sampled_test_dataset_df = pd.DataFrame(sampled_test_dataset)

# Save the sampled datasets to CSV files
print("Saving sampled datasets to CSV files...")
sampled_train_dataset_df.to_csv(train_dataset_name, index=False)
sampled_val_dataset_df.to_csv(val_dataset_name, index=False)
sampled_test_dataset_df.to_csv(test_dataset_name, index=False)
print("Sampled datasets saved successfully!")

# Load the sampled datasets from CSV files
print("Loading sampled datasets from CSV files...")
sampled_train_dataset = pd.read_csv(train_dataset_name, na_values="")
sampled_val_dataset = pd.read_csv(val_dataset_name, na_values="")
sampled_test_dataset = pd.read_csv(test_dataset_name, na_values="")
print("Sampled datasets loaded successfully!")

# Drop rows with missing values
print("Dropping rows with missing values...")
sampled_train_dataset = sampled_train_dataset.dropna()
sampled_val_dataset = sampled_val_dataset.dropna()
sampled_test_dataset = sampled_test_dataset.dropna()
print("Rows with missing values dropped successfully!")

# Save the cleaned datasets to CSV files
print("Saving cleaned datasets to CSV files...")
sampled_train_dataset.to_csv(train_dataset_name, index=False)
sampled_val_dataset.to_csv(val_dataset_name, index=False)
sampled_test_dataset.to_csv(test_dataset_name, index=False)
print("Cleaned datasets saved successfully!")