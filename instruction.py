import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import tkinter as tk
import os
import docx  # Library to handle Word documents

# Load data and create service names
data = pd.read_excel('dataset.xlsx')
texts = data['Topic'].tolist()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert_model.encode(texts, convert_to_tensor=True)
service_names = list(data['label'].astype('category').cat.categories)

# Load the trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("enhanced_trained_model_Correct")
model = DistilBertForSequenceClassification.from_pretrained("enhanced_trained_model_Correct")

# Instructions from dataset
instructions = data['Solution'].dropna().tolist()
instructions = [str(instruction) for instruction in instructions]
instruction_embeddings = sbert_model.encode(instructions, convert_to_tensor=True)

# Function to find top N similar Word files
def find_top_files(instruction_text, top_n=3):
    folder_path = 'Instructions'
    instruction_embedding = sbert_model.encode(instruction_text, convert_to_tensor=True)

    # Find top N most similar Word files
    similarities = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            doc_path = os.path.join(folder_path, filename)
            doc = docx.Document(doc_path)
            full_text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            doc_embedding = sbert_model.encode(full_text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(instruction_embedding, doc_embedding).item()
            similarities.append((score, filename))

    # Sort by similarity score and return top N files
    top_files = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
    return [filename for score, filename in top_files]

# Function to display buttons for top instruction files
def show_instruction_buttons(top_files):
    clear_instruction_buttons()  # Clear any existing buttons first
    
    # Create a button for each top file
    for idx, filename in enumerate(top_files):
        button = tk.Button(result_frame, text=filename, 
                           command=lambda f=filename: os.startfile(os.path.join('Instructions', f)),
                           relief=tk.FLAT)
        button.pack(pady=5)  # Add vertical padding between buttons


# Function to analyze the problem and display results
def analyze_problem(problem_text):
    try:
        # Tokenize and run through model
        inputs = tokenizer(problem_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        outputs = model(**inputs)
        softmaxed = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_probs, top_indices = torch.topk(softmaxed, k=3)

        # Extract top services and probabilities
        top_services = [(service_names[idx], prob.item() * 100) for idx, prob in zip(top_indices[0], top_probs[0])]

        # Find similar problems using SBERT embeddings
        query_embedding = sbert_model.encode(problem_text, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(query_embedding, text_embeddings)
        top_similarities, top_indices = torch.topk(cos_sim[0], k=min(5, len(texts)))
        top_problems = [(texts[idx.item()], round(sim.item() * 100, 2)) for idx, sim in zip(top_indices, top_similarities)]
    except Exception as e:
        clear_results()
        tk.Label(result_frame, text=f"Error during analysis: {e}").pack()
        return

    try:
        # Find the most similar brief instruction
        instruction_cos_sim = util.pytorch_cos_sim(query_embedding, instruction_embeddings)
        best_instruction_idx = torch.argmax(instruction_cos_sim).item()
        best_instruction = instructions[best_instruction_idx] if pd.notnull(instructions[best_instruction_idx]) else "No instruction found."
    except Exception as e:
        best_instruction = "Error retrieving instruction: " + str(e)

    clear_results()

    # Display service suggestions
    tk.Label(result_frame, text="Service suggestions:", font=("Arial", 14)).pack()
    for i, (service, prob) in enumerate(top_services[:3]):
        tk.Label(result_frame, text=f"{i+1}. {service}: {prob:.2f}%", font=("Arial", 12)).pack()

    # Display similar problems
    tk.Label(result_frame, text="\nMost similar problems:", font=("Arial", 14)).pack()
    for problem_idx, (problem, percentage) in enumerate(top_problems[:3]):  # Show only the top 3 similar problems
        dataset_index = top_indices[problem_idx].item()  # Get the dataset index for the problem
        tk.Label(result_frame, text=f"{dataset_index+1}. {problem} ({percentage}%)", font=("Arial", 12)).pack()

    # Display suggested instruction
    tk.Label(result_frame, text="\nSuggested instruction:", font=("Arial", 14)).pack()
    tk.Label(result_frame, text=f"{best_instruction}", font=("Arial", 12)).pack()

    # Button to find and show top files for detailed instructions
    detailed_button = tk.Button(result_frame, text="View Detailed Instructions", 
                                command=lambda: show_instruction_buttons(find_top_files(best_instruction, top_n=3)))
    detailed_button.pack()

# Clear previous results from the frame
def clear_results():
    for widget in result_frame.winfo_children():
        widget.destroy()

# Clear any existing instruction buttons
def clear_instruction_buttons():
    for widget in result_frame.winfo_children():
        if isinstance(widget, tk.Button) and widget.cget("text").startswith("Open Instruction"):
            widget.destroy()

# Tkinter GUI setup
window = tk.Tk()
window.title("Problem Analyzer")

label_problem = tk.Label(window, text="Enter your problem description:")
label_problem.pack()
entry_problem = tk.Entry(window, width=50)
entry_problem.pack()

button_analyze = tk.Button(window, text="Analyze", command=lambda: analyze_problem(entry_problem.get()))
button_analyze.pack()

label_result = tk.Label(window, text="Result:")
label_result.pack()

result_frame = tk.Frame(window)
result_frame.pack()

window.mainloop()
