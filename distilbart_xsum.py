import torch
from datasets import load_dataset
import os
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np

wandb.init(project="summarization_xsum")

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

train_dataset_name = "./datasets/500/sampled_train_dataset.csv"
val_dataset_name = "./datasets/500/sampled_val_dataset.csv"
test_dataset_name = "./datasets/500/sampled_test_dataset.csv"


# Check if the files exist
assert os.path.exists(train_dataset_name), f"Training dataset not found at {train_dataset_name}"
assert os.path.exists(val_dataset_name), f"Validation dataset not found at {val_dataset_name}"
assert os.path.exists(test_dataset_name), f"Test dataset not found at {test_dataset_name}"


print("All dataset files exist.")

# Convert the cleaned datasets to Hugging Face datasets
print("Converting datasets to Hugging Face datasets...")
sampled_train_dataset = load_dataset("csv", data_files=train_dataset_name)
sampled_val_dataset = load_dataset("csv", data_files=val_dataset_name)
sampled_test_dataset = load_dataset("csv", data_files=test_dataset_name)

sampled_train_dataset = sampled_train_dataset["train"]
sampled_val_dataset = sampled_val_dataset["train"]
sampled_test_dataset = sampled_test_dataset["train"]

print("Datasets converted successfully!")


# Load tokenizer
print("Loading tokenizer...")
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

# Load model for summarization
print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6")

print("Model loaded successfully!")

# Define training arguments
print("Defining training arguments...")

model_dir = "./distilbart-xsum-500-v1.2.7"

training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,       # output directory
    eval_strategy="epoch",                # evaluation strategy
    learning_rate=6e-6,                   # increased learning rate
    per_device_train_batch_size=4,        # increased batch size for training
    per_device_eval_batch_size=4,         # increased batch size for evaluation
    num_train_epochs=5,                   # reduced number of epochs
    weight_decay=0.05,                    # weight decay for regularization
    logging_dir="./logs",                 # directory for logs
    logging_steps=100,                    # log every 100 steps
    predict_with_generate=True,           # enables generate for predictions
    save_strategy="epoch",                # save model every epoch
    report_to=["wandb"],                    # enable wandb reports
)
print("Training arguments defined successfully!")

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
trainer.train()
print("Model trained successfully!")

# Evaluate the model
print("Evaluating the model...")
evaluation_results = trainer.evaluate()
print("Evaluation results:")
print(evaluation_results)

# Save the fine-tuned model and tokenizer
print("Saving the fine-tuned model and tokenizer...")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print("Model and tokenizer saved successfully!")

device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Lists to store scores and lengths
rouge1_precisions, rouge1_recalls, rouge1_fmeasures = [], [], []
rouge2_precisions, rouge2_recalls, rouge2_fmeasures = [], [], []
rougeL_precisions, rougeL_recalls, rougeL_fmeasures = [], [], []
bleu_scores = []
bertscore_precisions, bertscore_recalls, bertscore_f1s = [], [], []
generated_lengths, reference_lengths = [], []

# Smoothing function for BLEU score
smoothing = SmoothingFunction().method1

# Iterate over all examples in the test dataset
for i, row in enumerate(sampled_test_dataset):
    input_text = row["document"]
    reference_summary = row["summary"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, generated_summary)

    # Store ROUGE scores
    rouge1_precisions.append(scores['rouge1'].precision)
    rouge1_recalls.append(scores['rouge1'].recall)
    rouge1_fmeasures.append(scores['rouge1'].fmeasure)

    rouge2_precisions.append(scores['rouge2'].precision)
    rouge2_recalls.append(scores['rouge2'].recall)
    rouge2_fmeasures.append(scores['rouge2'].fmeasure)

    rougeL_precisions.append(scores['rougeL'].precision)
    rougeL_recalls.append(scores['rougeL'].recall)
    rougeL_fmeasures.append(scores['rougeL'].fmeasure)

    # Calculate BLEU score
    reference_tokens = word_tokenize(reference_summary)
    generated_tokens = word_tokenize(generated_summary)
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    bleu_scores.append(bleu_score)

    # Calculate BERTScore
    P, R, F1 = bert_score([generated_summary], [reference_summary], lang="en", verbose=True)
    bertscore_precisions.append(P.mean().item())
    bertscore_recalls.append(R.mean().item())
    bertscore_f1s.append(F1.mean().item())

    # Store lengths of generated and reference summaries
    generated_lengths.append(len(generated_summary.split()))
    reference_lengths.append(len(reference_summary.split()))

    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(sampled_test_dataset)} examples")

avg_rouge1_precision = np.mean(rouge1_precisions)
avg_rouge1_recall = np.mean(rouge1_recalls)
avg_rouge1_fmeasure = np.mean(rouge1_fmeasures)

avg_rouge2_precision = np.mean(rouge2_precisions)
avg_rouge2_recall = np.mean(rouge2_recalls)
avg_rouge2_fmeasure = np.mean(rouge2_fmeasures)

avg_rougeL_precision = np.mean(rougeL_precisions)
avg_rougeL_recall = np.mean(rougeL_recalls)
avg_rougeL_fmeasure = np.mean(rougeL_fmeasures)

# Define custom evaluation metrics
metrics = {
    "rouge1_precision": avg_rouge1_precision,
    "rouge1_recall": avg_rouge1_recall,
    "rouge1_fmeasure": avg_rouge1_fmeasure,
    "rouge2_precision": avg_rouge2_precision,
    "rouge2_recall": avg_rouge2_recall,
    "rouge2_fmeasure": avg_rouge2_fmeasure,
    "rougeL_precision": avg_rougeL_precision,
    "rougeL_recall": avg_rougeL_recall,
    "rougeL_fmeasure": avg_rougeL_fmeasure,
    "bleu_score": np.mean(bleu_scores),
    "bertscore_precision": np.mean(bertscore_precisions),
    "bertscore_recall": np.mean(bertscore_recalls),
    "bertscore_f1": np.mean(bertscore_f1s)
}

print("Evaluation metrics:")
print(metrics)


# Log the metrics to Weights and Biases
wandb.log(metrics)

wandb.finish()
