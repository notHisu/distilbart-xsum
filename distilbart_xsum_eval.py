import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from evaluate import load

model_name = "./distilbart-xsum-300"
tokenizer_name = "./distilbart-xsum-300"

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
model.to(device)
print(f"Using device: {device}")

# Set the model to evaluation mode
model.eval()
print("Model loaded successfully")

val_dataset_name = "sampled_val_dataset.csv"

# Load the validation dataset
print("Loading validation dataset...")
val_dataset = load_dataset("csv", data_files=val_dataset_name)["train"]
print("Validation dataset loaded successfully")

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

# Tokenize the validation dataset
print("Tokenizing the validation dataset...")
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
print("Validation dataset tokenized successfully")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",          # output directory
    per_device_eval_batch_size=8,    # batch size for evaluation
    predict_with_generate=True       # enables generate for predictions
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,                              # the instantiated 🤗 Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    eval_dataset=tokenized_val_dataset,       # evaluation dataset
    processing_class=tokenizer                # tokenizer
)

# Evaluate the model
print("Evaluating the model...")
evaluation_results = trainer.evaluate()

# Load the ROUGE metric
rouge = load("rouge")

# Generate predictions
print("Generating predictions...")
predictions, labels, _ = trainer.predict(tokenized_val_dataset)
decoded_preds = trainer.processing_class.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = trainer.processing_class.batch_decode(labels, skip_special_tokens=True)

# Compute ROUGE scores
print("Computing ROUGE scores...")
rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
print("ROUGE results:")
print(rouge_results)

# Load the BLEU metric
bleu = load("bleu")

# Compute BLEU scores
print("Computing BLEU scores...")
bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
print("BLEU results:")
print(bleu_results)