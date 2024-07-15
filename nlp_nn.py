import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import json
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

# Check for GPU availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicon GPU) is available. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

# Data Generation

def generate_network():
    layers = []
    input_shape = [random.randint(1, 256) for _ in range(random.randint(1, 3))]
    current_shape = input_shape.copy()

    for _ in range(random.randint(1, 10)):
        layer_type = random.choice(['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'LSTM', 'GRU'])
        if layer_type == 'Linear':
            in_features = current_shape[-1]
            out_features = random.randint(1, 256)
            layers.append(f"nn.Linear(in_features={in_features}, out_features={out_features})")
            current_shape = current_shape[:-1] + [out_features]
        elif layer_type in ['Conv1d', 'Conv2d', 'Conv3d']:
            in_channels = current_shape[0]
            out_channels = random.randint(1, 64)
            kernel_size = random.randint(1, 5)
            layers.append(f"nn.{layer_type}(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size})")
            current_shape[0] = out_channels
        elif layer_type in ['LSTM', 'GRU']:
            input_size = current_shape[-1]
            hidden_size = random.randint(1, 128)
            layers.append(f"nn.{layer_type}(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)")
            current_shape[-1] = hidden_size

    return input_shape, current_shape, layers

def generate_description(input_shape, output_shape):
    return f"This neural network takes an input of shape {input_shape} and produces an output of shape {output_shape}."

def generate_data(num_samples, time_limit):
    data = []
    start_time = time.time()
    for _ in tqdm(range(num_samples), desc="Generating data"):
        if time.time() - start_time > time_limit:
            break
        input_shape, output_shape, layers = generate_network()
        network = "nn.Sequential(\n    " + ",\n    ".join(layers) + "\n)"
        description = generate_description(input_shape, output_shape)
        data.append({"network": network, "description": description})
    return data

# Data Analysis and Visualization

def analyze_data(data):
    layer_types = []
    input_shapes = []
    output_shapes = []

    for item in data:
        network = item['network']
        layer_types.extend(re.findall(r'nn\.(\w+)', network))
        input_shape, output_shape = extract_shapes(item['description'])
        input_shapes.append(eval(input_shape))
        output_shapes.append(eval(output_shape))

    return layer_types, input_shapes, output_shapes

def visualize_data(layer_types, input_shapes, output_shapes):
    plt.figure(figsize=(12, 6))
    sns.countplot(y=layer_types)
    plt.title('Distribution of Layer Types')
    plt.tight_layout()
    plt.savefig('layer_types_distribution.png')
    plt.close()

    input_dims = [len(shape) for shape in input_shapes]
    plt.figure(figsize=(8, 6))
    sns.histplot(input_dims, bins=range(1, max(input_dims)+2), kde=True)
    plt.title('Distribution of Input Shape Dimensions')
    plt.xlabel('Number of Dimensions')
    plt.tight_layout()
    plt.savefig('input_shape_dimensions.png')
    plt.close()

    output_dims = [len(shape) for shape in output_shapes]
    plt.figure(figsize=(8, 6))
    sns.histplot(output_dims, bins=range(1, max(output_dims)+2), kde=True)
    plt.title('Distribution of Output Shape Dimensions')
    plt.xlabel('Number of Dimensions')
    plt.tight_layout()
    plt.savefig('output_shape_dimensions.png')
    plt.close()

    input_sizes = [np.prod(shape) for shape in input_shapes]
    plt.figure(figsize=(10, 6))
    sns.histplot(input_sizes, bins=30, kde=True)
    plt.title('Distribution of Input Sizes')
    plt.xlabel('Input Size (Total Number of Elements)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('input_sizes_distribution.png')
    plt.close()

    output_sizes = [np.prod(shape) for shape in output_shapes]
    plt.figure(figsize=(10, 6))
    sns.histplot(output_sizes, bins=30, kde=True)
    plt.title('Distribution of Output Sizes')
    plt.xlabel('Output Size (Total Number of Elements)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('output_sizes_distribution.png')
    plt.close()

# Model and Dataset

class NeuralNetworkDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.encode_plus(
            item['network'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer.encode_plus(
            item['description'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

# Training and Evaluation

def train(model, train_dataloader, optimizer, device, max_time):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch in tqdm(train_dataloader, desc="Training"):
        if time.time() - start_time > max_time:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_dataloader)

def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(eval_dataloader)

def generate_description_for_network(model, tokenizer, network_str, device):
    input_ids = tokenizer.encode(network_str, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_shapes(description):
    shapes = re.findall(r'\[.*?\]', description)
    return shapes[0] if shapes else None, shapes[-1] if shapes else None

def evaluate_model(model, tokenizer, test_data, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            input_text = item['network']
            true_description = item['description']
            
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            output = model.generate(input_ids, max_length=100)
            predicted_description = tokenizer.decode(output[0], skip_special_tokens=True)
            
            true_input, true_output = extract_shapes(true_description)
            pred_input, pred_output = extract_shapes(predicted_description)
            
            true_labels.append((true_input, true_output))
            predicted_labels.append((pred_input, pred_output))

    return true_labels, predicted_labels

def calculate_metrics(true_labels, predicted_labels):
    correct_input = sum(t[0] == p[0] for t, p in zip(true_labels, predicted_labels))
    correct_output = sum(t[1] == p[1] for t, p in zip(true_labels, predicted_labels))
    correct_both = sum(t == p for t, p in zip(true_labels, predicted_labels))
    
    total = len(true_labels)
    
    accuracy_input = correct_input / total
    accuracy_output = correct_output / total
    accuracy_both = correct_both / total
    
    precision = correct_both / total
    recall = correct_both / total
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'Accuracy (Input Shape)': accuracy_input,
        'Accuracy (Output Shape)': accuracy_output,
        'Accuracy (Both Shapes)': accuracy_both,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def plot_confusion_matrix(true_labels, predicted_labels):
    true_simplified = ['Correct' if t[0] == p[0] and t[1] == p[1] else 'Incorrect' for t, p in zip(true_labels, predicted_labels)]
    pred_simplified = ['Correct' if t[0] == p[0] and t[1] == p[1] else 'Incorrect' for t, p in zip(true_labels, predicted_labels)]
    
    cm = confusion_matrix(true_simplified, pred_simplified, labels=['Correct', 'Incorrect'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Correct', 'Incorrect'], yticklabels=['Correct', 'Incorrect'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_test_results(true_labels, predicted_labels):
    correct_predictions = [t == p for t, p in zip(true_labels, predicted_labels)]
    incorrect_predictions = [t != p for t, p in zip(true_labels, predicted_labels)]
    
    correct_samples = [t for t, c in zip(true_labels, correct_predictions) if c]
    incorrect_samples = [t for t, c in zip(true_labels, incorrect_predictions) if c]
    
    print("\nTest Data Analysis:")
    print(f"Total samples: {len(true_labels)}")
    print(f"Correct predictions: {sum(correct_predictions)}")
    print(f"Incorrect predictions: {sum(incorrect_predictions)}")
    
    print("\nMost common correct predictions:")
    print(Counter(correct_samples).most_common(5))
    
    print("\nMost common incorrect predictions:")
    print(Counter(incorrect_samples).most_common(5))

def error_analysis(true_labels, predicted_labels):
    errors = [(t, p) for t, p in zip(true_labels, predicted_labels) if t != p]
    
    print("\nError Analysis:")
    print(f"Total errors: {len(errors)}")
    
    input_shape_errors = sum(t[0] != p[0] for t, p in errors)
    output_shape_errors = sum(t[1] != p[1] for t, p in errors)
    
    print(f"Input shape errors: {input_shape_errors}")
    print(f"Output shape errors: {output_shape_errors}")
    
    print("\nSample of errors:")
    for true, pred in errors[:5]:
        print(f"True: {true}, Predicted: {pred}")

# Main execution
def main():
    print("Starting the Comprehensive GPU Neural Network Description Generator...")
    
    # Data Generation
    print("Generating data...")
    train_data = generate_data(100000, 7200)  # 2 hours = 7200 seconds
    eval_data = generate_data(10000, 600)  # 10 minutes for eval data

    print(f"Generated {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # Save data to files
    with open('train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open('eval_data.json', 'w') as f:
        json.dump(eval_data, f)

    print("Data generation completed and saved.")

    # Data Analysis and Visualization
    print("Analyzing and visualizing data...")
    layer_types, input_shapes, output_shapes = analyze_data(train_data)
    visualize_data(layer_types, input_shapes, output_shapes)

    # Model Training
    print("Preparing model and datasets...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_dataset = NeuralNetworkDataset(train_data, tokenizer, max_length=256)
    eval_dataset = NeuralNetworkDataset(eval_data, tokenizer, max_length=256)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)

    print("Starting training...")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 10
    max_train_time = 7200  # 2 hours
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, device, max_train_time - (time.time() - start_time))
        eval_loss = evaluate(model, eval_dataloader, device)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Eval loss: {eval_loss:.4f}")
        
        if time.time() - start_time > max_train_time:
            print("Training time limit reached. Stopping training.")
            break

    # Save the model
    model.save_pretrained("neural_network_description_model")
    tokenizer.save_pretrained("neural_network_description_model")

    print("Training completed and model saved.")

    # Model Evaluation
    print("Evaluating the model...")
    loaded_model = T5ForConditionalGeneration.from_pretrained("neural_network_description_model").to(device)
    loaded_tokenizer = T5Tokenizer.from_pretrained("neural_network_description_model")

    true_labels, predicted_labels = evaluate_model(loaded_model, loaded_tokenizer, eval_data, device)

    # Calculate and display metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels)

    # Test Data Analysis
    analyze_test_results(true_labels, predicted_labels)

    # Error Analysis
    error_analysis(true_labels, predicted_labels)

    # Evaluation on 5 different unique neural network architectures
    print("\nEvaluating on 5 different unique neural network architectures...")
    
    test_networks = [
        """nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32768, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )""",
        """nn.Sequential(
            nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )""",
        """nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )""",
        """nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=10)
        )""",
        """nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )"""
    ]

    true_shapes = [
        ([3, 224, 224], [10]),
        ([32, 100], [1]),
        ([1, 500], [10]),
        ([784], [10]),
        ([1, 32, 32, 32], [1])
    ]

    test_true_labels = []
    test_predicted_labels = []

    for i, network in enumerate(test_networks):
        print(f"\nTesting network {i+1}:")
        print(network)
        generated_description = generate_description_for_network(loaded_model, loaded_tokenizer, network, device)
        print("Generated description:", generated_description)
        pred_input, pred_output = extract_shapes(generated_description)
        print(f"Predicted shapes: Input {pred_input}, Output {pred_output}")
        print(f"True shapes: Input {true_shapes[i][0]}, Output {true_shapes[i][1]}")
        
        test_true_labels.append((str(true_shapes[i][0]), str(true_shapes[i][1])))
        test_predicted_labels.append((pred_input, pred_output))

    # Calculate metrics for the 5 test networks
    test_metrics = calculate_metrics(test_true_labels, test_predicted_labels)
    print("\nTest Network Performance Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Detailed analysis of test results
    print("\nDetailed analysis of test results:")
    for i, (true, pred) in enumerate(zip(test_true_labels, test_predicted_labels)):
        print(f"\nNetwork {i+1}:")
        print(f"True: Input {true[0]}, Output {true[1]}")
        print(f"Predicted: Input {pred[0]}, Output {pred[1]}")
        print(f"Input shape correct: {true[0] == pred[0]}")
        print(f"Output shape correct: {true[1] == pred[1]}")
        print(f"Both shapes correct: {true == pred}")

    print("\nComprehensive GPU Neural Network Description Generator process completed.")

if __name__ == "__main__":
    main()