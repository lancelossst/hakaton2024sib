import os
import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from docx import Document

# Load dataset with requests and labels
data = pd.read_excel('dataset.xlsx')
requests = data['Topic'].tolist()
services = data['label'].astype('category').cat.codes.tolist()
request_ids = data['№'].tolist()  # Assuming 'RequestID' column has unique request numbers
labels_to_services = data['label'].astype('category').cat.categories

# Load Tokenizer and Embedding Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and process detailed Word files
detailed_texts = []
detailed_files_path = 'Instructions'  # Path to the folder with Word files

for filename in os.listdir(detailed_files_path):
    if filename.endswith(".docx"):
        doc = Document(os.path.join(detailed_files_path, filename))
        detailed_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])  # Combine all paragraphs
        detailed_texts.append((filename, detailed_text))  # Store filename and content

# Convert detailed texts into embeddings for comparison
detailed_embeddings = sbert_model.encode([text for _, text in detailed_texts])

# Convert problem requests into embeddings for comparison
request_embeddings = sbert_model.encode(requests)

# Find best match for each problem request
request_to_instruction = {}
for i, request_embedding in enumerate(request_embeddings):
    similarities = np.dot(detailed_embeddings, request_embedding)
    best_match_idx = np.argmax(similarities)
    best_match_filename, best_match_text = detailed_texts[best_match_idx]
    request_to_instruction[request_ids[i]] = best_match_filename  # Map request ID to the matched file name

# Custom Dataset with Context Weights
class ProblemDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.context_phrases = ["doesn't work", "not working", "won't start", "is broken", "stopped working", "malfunction"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        item = {key: val.flatten() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Adjusting weights based on presence of context phrases
        item["weights"] = torch.ones(item['input_ids'].shape)  # Initialize weights as ones
        for phrase in self.context_phrases:
            phrase_tokens = tokenizer(phrase, return_tensors="pt")["input_ids"][0][1:-1]
            for i in range(len(item['input_ids']) - len(phrase_tokens)):
                if torch.equal(item['input_ids'][i:i+len(phrase_tokens)], phrase_tokens):
                    # Increase weight for hardware-related phrases
                    item["weights"][i:i+len(phrase_tokens)] = 2.0
                    # Apply increased weights for adjacent words
                    if i > 0:
                        item["weights"][i - 1] = 1.5
                    if i + len(phrase_tokens) < len(item["weights"]):
                        item["weights"][i + len(phrase_tokens)] = 1.5
        return item

# Train-Validation Split
train_texts, val_texts, train_services, val_services = train_test_split(requests, services, test_size=0.2, random_state=42)
train_dataset = ProblemDataset(train_texts, train_services)
val_dataset = ProblemDataset(val_texts, val_services)

# Define Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels_to_services))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array(list(set(services))), y=services)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply class weights to the CrossEntropyLoss
        loss_fn = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Training Arguments
training_args = TrainingArguments(
    output_dir='./enhanced_results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()
model.save_pretrained("enhanced_trained_model_Correct")
tokenizer.save_pretrained("enhanced_trained_model_Correct")
# # Output the mapping of request IDs to detailed instruction files
# print("Request to Instruction Mapping:")
# for request_id, instruction_file in request_to_instruction.items():
#     print(f"Request ID {request_id}: Matched Instruction File - {instruction_file}")
