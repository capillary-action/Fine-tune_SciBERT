# Fine-tune_SciBERT

   - Downloads NLTK stopwords and stemmer data.
   - Loads the SciBERT tokenizer and model.
   - Defines the training arguments for the model.
   - Reads the contents of all files in the "output_text" directory and appends them to a list.
   - Merges the contents of the list into a single string.
   - Writes the merged string to a file called "formatted_output.txt".
   - Creates a TextDataset using the "formatted_output.txt" file and the SciBERT tokenizer.
   - Defines a data collator for language modeling.
   - Creates a Trainer object using the SciBERT model, the training arguments, the data collator, and the TextDataset.
   - Trains the model using the Trainer object.
