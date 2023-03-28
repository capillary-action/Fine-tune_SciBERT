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
#model.eval()

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
directory = '/home/pepesilvia/scibert/output_text'

# Loop through all files in the directory and read their contents into formatted_text_data
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as file:
        formatted_text_data.append(file.read())

# Define the stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Tokenize and split the formatted text data into sequences of length 512
encoded_data = tokenizer.batch_encode_plus(
    formatted_text_data,
    max_length=512,
    truncation=True,
    padding='max_length'
)

# Create a PyTorch Dataset from the encoded data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx]}

    def __len__(self):
        return len(self.input_ids)

dataset = TextDataset(encoded_data['input_ids'], encoded_data['attention_mask'])




#print("Dataset length:", len(dataset))



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

# Train the model
trainer.train()

# Generate text from the trained model
input_ids = tokenizer.encode("Some input text")
input_ids_tensor = torch.tensor([input_ids])
generated_sequences = model.generate(
    input_ids=input_ids_tensor,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Print the generated text
print(tokenizer.decode(generated_sequences[0]))


