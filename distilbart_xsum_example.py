from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import load_dataset
import random

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

model_name = "./distilbart-xsum-300"
tokenizer_name = "./distilbart-xsum-300"

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

test_dataset_name = "./datasets/300/sampled_test_dataset.csv"

# Load test dataset
sampled_test_dataset = load_dataset("csv", data_files=test_dataset_name)
sampled_test_dataset = sampled_test_dataset["train"]

# Take a random row from the test dataset
random_index = random.randint(0, len(sampled_test_dataset) - 1)
random_row = sampled_test_dataset[random_index]

# Store the document as input_text and the summary as result
input_text = random_row["document"]
result = random_row["summary"]

print("Input text:", input_text)
print("Reference summary:", result)

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Generate summary
summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated summary:", summary)

# Calculate ROUGE scores
from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(summary, result)
print("ROUGE scores:", scores)

# Calculate BLEU scores
from nltk.translate.bleu_score import sentence_bleu

reference = result.split()
candidate = summary.split()
bleu_score = sentence_bleu([reference], candidate)
print("BLEU score:", bleu_score)


