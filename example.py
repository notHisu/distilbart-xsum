from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import load_dataset
import random

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
# print(f"Using device: {device}")

# Define a list of models to choose from
models = [
    "./distilbart-xsum-500-v1",
    "./distilbart-xsum-500-v2",
    "./distilbart-xsum-500-v3",
    "./distilbart-xsum-500-v4",
    "./distilbart-xsum-500-v5",
    "./distilbart-xsum-500-v1.1",
    "./distilbart-xsum-500-v1.2",
    "./distilbart-xsum-500-v1.3",
    "./distilbart-xsum-500-v1.4",
    "./distilbart-xsum-500-v1.5",
    "./distilbart-xsum-500-v1.2.1",
    "./distilbart-xsum-500-v1.2.2",
    "./distilbart-xsum-500-v1.2.3",
    "./distilbart-xsum-500-v1.2.4",
    "./distilbart-xsum-500-v1.2.5",
    "./distilbart-xsum-500-v1.2.6",
    "./distilbart-xsum-500-v1.2.7",
]


# Load test dataset
test_dataset_name = "./datasets/500/sampled_test_dataset.csv"
sampled_test_dataset = load_dataset("csv", data_files=test_dataset_name)
sampled_test_dataset = sampled_test_dataset["train"]

# Take a random row from the test dataset
random_index = random.randint(0, len(sampled_test_dataset) - 1)
random_row = sampled_test_dataset[45]

# Store the document as input_text and the summary as result
input_text = random_row["document"]
result = random_row["summary"]

print("Input text:", input_text + "\n")
print("Reference summary:", result + "\n")


# For each model, load the model and tokenizer
for model_name in models:
    # print(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model loaded successfully: {model_name}")

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

    # Decode and print the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Generated summary:", summary + "\n")


# input_text = "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don't think I'll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I'll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say 'kid star goes off the rails,'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: \"I just think I'm going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."

# result = "Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday. Young actor says he has no plans to fritter his cash away. Radcliffe's earnings from first five Potter films have been held in trust fund."
