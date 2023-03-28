import os
import re
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from PIL import Image as PILImage
from wand.image import Image as WandImage
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the NLTK stopwords and stemmer data
nltk.download('stopwords')
nltk.download('punkt')

# Load the SciBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')

# Set the model to training mode
model.train()

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=2e-5,
    overwrite_output_dir=True,
    do_train=True
)

# Create a list to store the formatted text data
formatted_text_data = []

# Define the path to the local directory containing the formatted text data
directory = 'output_text'

# Loop through all files in the directory and read their contents into formatted_text_data
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as file:
        formatted_text_data.append(file.read())

# Define the stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Merge the formatted text data into a single string
merged_text_data = "\n".join(formatted_text_data)

# Save the merged text data to a text file
with open('formatted_output.txt', mode='w') as file:
    file.write(merged_text_data)

# Load the merged text data from the text file
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='formatted_output.txt',
    block_size=512
)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Create a Trainer object and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)
#trainer.train()
