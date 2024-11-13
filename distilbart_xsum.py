import torch
from datasets import load_dataset
import os

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

train_dataset_name = "./datasets/300/sampled_train_dataset.csv"
val_dataset_name = "./datasets/300/sampled_val_dataset.csv"

# Check if the files exist
assert os.path.exists(train_dataset_name), f"Training dataset not found at {train_dataset_name}"
assert os.path.exists(val_dataset_name), f"Validation dataset not found at {val_dataset_name}"

print("Both dataset files exist.")

# Convert the cleaned datasets to Hugging Face datasets
print("Converting datasets to Hugging Face datasets...")
sampled_train_dataset = load_dataset("csv", data_files=train_dataset_name)
sampled_val_dataset = load_dataset("csv", data_files=val_dataset_name)
print("Datasets converted successfully!")

sampled_train_dataset = sampled_train_dataset["train"]
sampled_val_dataset = sampled_val_dataset["train"]
print("Datasets converted successfully!")

from transformers import AutoTokenizer

# Load DistilBART tokenizer
print("Loading DistilBART tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")
print("Tokenizer loaded successfully!")

# Define the tokenization function
def tokenize_function(examples):
    try:
        # Tokenize the inputs and outputs
        model_inputs = tokenizer(examples["document"], max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["summary"], max_length=150, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except Exception as e:
        print(f"Error processing example: {examples}")
        print(f"Error message: {e}")
        raise e

# Apply tokenization to the preprocessed datasets
print("Tokenizing the dataset...")
tokenized_train_dataset = sampled_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = sampled_val_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized successfully!")

from transformers import AutoModelForSeq2SeqLM

# Load DistilBART model for summarization
print("Loading DistilBART model...")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6")
print("Model loaded successfully!")

from transformers import Seq2SeqTrainingArguments

# Define training arguments
print("Defining training arguments...")
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./distilbart-xsum",       # output directory
#     evaluation_strategy="epoch",          # evaluation strategy
#     learning_rate=2e-5,                   # learning rate
#     per_device_train_batch_size=4,        # batch size for training
#     per_device_eval_batch_size=4,         # batch size for evaluation
#     num_train_epochs=3,                   # number of epochs
#     weight_decay=0.01,                    # weight decay for regularization
#     logging_dir="./logs",                 # directory for logs
#     predict_with_generate=True            # enables generate for predictions
# )
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./distilbart-xsum-300",       # output directory
#     eval_strategy="epoch",                # evaluation strategy
#     learning_rate=5e-5,                   # increased learning rate
#     per_device_train_batch_size=4,        # increased batch size for training
#     per_device_eval_batch_size=8,         # increased batch size for evaluation
#     num_train_epochs=5,                   # reduced number of epochs
#     weight_decay=0.01,                    # weight decay for regularization
#     logging_dir="./logs",                 # directory for logs
#     logging_steps=500,                    # log every 500 steps
#     predict_with_generate=True,           # enables generate for predictions
#     save_strategy="epoch",                # save model every epoch
# )

model_name = "./distilbart-xsum-300-v4"

training_args = Seq2SeqTrainingArguments(
    output_dir=model_name,       # output directory
    eval_strategy="epoch",                # evaluation strategy
    learning_rate=3e-5,                   # increased learning rate
    per_device_train_batch_size=4,        # increased batch size for training
    per_device_eval_batch_size=8,         # increased batch size for evaluation
    num_train_epochs=40,                   # reduced number of epochs
    weight_decay=0.01,                    # weight decay for regularization
    logging_dir="./logs",                 # directory for logs
    logging_steps=500,                    # log every 500 steps
    predict_with_generate=True,           # enables generate for predictions
    save_strategy="epoch",                # save model every epoch
)
print("Training arguments defined successfully!")

from transformers import Seq2SeqTrainer

# Initialize Trainer
print("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,                              # model to train
    args=training_args,                       # training arguments
    train_dataset=tokenized_train_dataset,    # training dataset
    eval_dataset=tokenized_val_dataset,       # validation dataset
    processing_class=tokenizer                # tokenizer
)
print("Trainer initialized successfully!")

# Start training
print("Training the model...")
# checkpoint_path = "./distilbart-xsum/checkpoint-875"
# trainer.train(resume_from_checkpoint=checkpoint_path)
trainer.train()
print("Model trained successfully!")

# Evaluate the model
print("Evaluating the model...")
evaluation_results = trainer.evaluate()
print("Evaluation results:")
print(evaluation_results)

# Save the fine-tuned model and tokenizer
print("Saving the fine-tuned model and tokenizer...")
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
print("Model and tokenizer saved successfully!")

