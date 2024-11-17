import pandas as pd

# Define strata for stratified sampling
bins = [0, 500, 1500, float("inf")]
labels = ["short", "medium", "long"]

# Function to count samples for each stratum
def count_samples_per_stratum(dataset, bins, labels):
    strata = pd.cut(dataset['document'].str.len(), bins=bins, labels=labels, right=True)
    counts = strata.value_counts().sort_index()
    return counts

# Load the sampled datasets from CSV files
print("Loading sampled datasets from CSV files...")
train_dataset_name = "./datasets/500/sampled_train_dataset.csv"
val_dataset_name = "./datasets/500/sampled_val_dataset.csv"
test_dataset_name = "./datasets/500/sampled_test_dataset.csv"

sampled_train_dataset = pd.read_csv(train_dataset_name, na_values="")
sampled_val_dataset = pd.read_csv(val_dataset_name, na_values="")
sampled_test_dataset = pd.read_csv(test_dataset_name, na_values="")
print("Sampled datasets loaded successfully!")

# Count samples for each stratum
print("Counting samples for each stratum in the training dataset...")
train_counts = count_samples_per_stratum(sampled_train_dataset, bins, labels)
print("Training dataset sample counts per stratum:")
print(train_counts)

print("Counting samples for each stratum in the validation dataset...")
val_counts = count_samples_per_stratum(sampled_val_dataset, bins, labels)
print("Validation dataset sample counts per stratum:")
print(val_counts)

print("Counting samples for each stratum in the test dataset...")
test_counts = count_samples_per_stratum(sampled_test_dataset, bins, labels)
print("Test dataset sample counts per stratum:")
print(test_counts)