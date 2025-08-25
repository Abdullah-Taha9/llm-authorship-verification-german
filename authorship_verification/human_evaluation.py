#!/usr/bin/env python3
"""
Human evaluation functionality for authorship verification.
Converted and adapted from the original human evaluation GUI.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Third-party
from datasets import load_dataset, Dataset
import portalocker

# Stdlib GUI
import tkinter as tk
from tkinter import ttk, messagebox


@dataclass
class PairFields:
    """Data structure to hold field mappings for text pairs."""
    left_key: str
    right_key: str
    label_key: Optional[str]
    id_key: Optional[str]
    lang_key: Optional[str]


class HumanEvaluationGUI:
    """GUI for human evaluation of authorship verification."""
    
    def __init__(self, dataset, output_file: str, language: str = "en"):
        self.dataset = dataset
        self.output_file = output_file
        self.language = language
        self.current_index = 0
        self.results = []
        
        # Load existing results if file exists
        self.load_existing_results()
        
        # Setup GUI
        self.setup_gui()
        
    def load_existing_results(self):
        """Load existing results from CSV file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.results = list(reader)
                print(f"Loaded {len(self.results)} existing evaluations")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.results = []
        else:
            self.results = []
    
    def setup_gui(self):
        """Setup the GUI components."""
        self.root = tk.Tk()
        self.root.title(f"Human Authorship Verification Evaluation - {self.language.upper()}")
        self.root.geometry("1000x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress info
        self.progress_var = tk.StringVar()
        self.update_progress()
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var, font=("Arial", 12, "bold"))
        progress_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Text displays
        ttk.Label(main_frame, text="Text 1:", font=("Arial", 14, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.text1_display = tk.Text(main_frame, height=8, width=60, wrap=tk.WORD, font=("Arial", 11))
        self.text1_display.grid(row=2, column=0, padx=(0, 10), pady=(0, 20))
        
        ttk.Label(main_frame, text="Text 2:", font=("Arial", 14, "bold")).grid(row=1, column=1, sticky=tk.W, pady=(0, 5))
        self.text2_display = tk.Text(main_frame, height=8, width=60, wrap=tk.WORD, font=("Arial", 11))
        self.text2_display.grid(row=2, column=1, pady=(0, 20))
        
        # Question
        question_label = ttk.Label(main_frame, 
                                 text="Do you think these texts were written by the same author?",
                                 font=("Arial", 14, "bold"))
        question_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        # Slider for certainty
        ttk.Label(main_frame, text="Your certainty (0 = Different Author, 1 = Same Author):", 
                 font=("Arial", 12)).grid(row=4, column=0, columnspan=2, pady=(0, 5))
        
        self.certainty_var = tk.DoubleVar()
        self.certainty_slider = ttk.Scale(main_frame, from_=0.0, to=1.0, 
                                        variable=self.certainty_var, orient=tk.HORIZONTAL, length=400)
        self.certainty_slider.grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        # Value display
        self.value_var = tk.StringVar()
        self.certainty_var.trace('w', self.update_slider_value)
        value_label = ttk.Label(main_frame, textvariable=self.value_var, font=("Arial", 12))
        value_label.grid(row=6, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(button_frame, text="Previous", command=self.previous_pair).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Submit & Next", command=self.submit_and_next).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Skip", command=self.skip_pair).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save & Exit", command=self.save_and_exit).pack(side=tk.LEFT)
        
        # Load first pair
        self.load_current_pair()
        
    def update_progress(self):
        """Update progress display."""
        total = len(self.dataset)
        current = self.current_index + 1
        completed = len(self.results)
        self.progress_var.set(f"Pair {current}/{total} | Completed: {completed}")
        
    def update_slider_value(self, *args):
        """Update slider value display."""
        value = self.certainty_var.get()
        self.value_var.set(f"Value: {value:.2f}")
        
    def load_current_pair(self):
        """Load the current text pair into the display."""
        if self.current_index >= len(self.dataset):
            messagebox.showinfo("Complete", "All pairs have been evaluated!")
            return
            
        item = self.dataset[self.current_index]
        
        # Extract texts
        text1 = item["review_1"]["review_body"] if "review_1" in item else item.get("text1", "")
        text2 = item["review_2"]["review_body"] if "review_2" in item else item.get("text2", "")
        
        # Clear and update text displays
        self.text1_display.delete(1.0, tk.END)
        self.text1_display.insert(1.0, text1)
        
        self.text2_display.delete(1.0, tk.END)
        self.text2_display.insert(1.0, text2)
        
        # Reset slider
        self.certainty_var.set(0.5)
        
        # Update progress
        self.update_progress()
        
    def submit_and_next(self):
        """Submit current evaluation and move to next pair."""
        if self.current_index >= len(self.dataset):
            return
            
        item = self.dataset[self.current_index]
        
        # Get evaluation data
        certainty = self.certainty_var.get()
        
        # Extract additional info
        text1 = item["review_1"]["review_body"] if "review_1" in item else item.get("text1", "")
        text2 = item["review_2"]["review_body"] if "review_2" in item else item.get("text2", "")
        true_label = item.get("label", None)
        
        # Create result record
        result = {
            "pair_index": self.current_index,
            "slider_value": certainty,
            "true_label": true_label,
            "language": self.language,
            "left_text": text1,
            "right_text": text2,
            "left_len": len(text1),
            "right_len": len(text2),
            "timestamp": time.time()
        }
        
        # Check if this pair was already evaluated
        existing_idx = None
        for i, existing in enumerate(self.results):
            if int(existing.get("pair_index", -1)) == self.current_index:
                existing_idx = i
                break
                
        if existing_idx is not None:
            self.results[existing_idx] = result
        else:
            self.results.append(result)
            
        # Save immediately
        self.save_results()
        
        # Move to next
        self.current_index += 1
        self.load_current_pair()
        
    def previous_pair(self):
        """Go to previous pair."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()
            
    def skip_pair(self):
        """Skip current pair without evaluation."""
        self.current_index += 1
        self.load_current_pair()
        
    def save_results(self):
        """Save results to CSV file."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                if self.results:
                    fieldnames = self.results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)
            print(f"Saved {len(self.results)} evaluations to {self.output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")
            
    def save_and_exit(self):
        """Save results and exit application."""
        self.save_results()
        self.root.quit()
        
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def run_human_evaluation(config_path: str):
    """Run human evaluation based on configuration."""
    import yaml
    import sys
    import os
    
    # Add data directory to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from data_loaders import get_data_loader
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    dataset = get_data_loader(config)
    
    # Create output filename using consistent naming pattern
    language = config["dataset_language"]
    output_file = f"results/human_evaluation/human_evaluation_{language}.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create and run GUI
    gui = HumanEvaluationGUI(dataset, output_file, language)
    gui.run()


def analyze_human_evaluation(csv_file: str, output_dir: str = "results/human_evaluation"):
    """Analyze human evaluation results."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import matplotlib.pyplot as plt
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Convert slider values to binary predictions
    threshold = 0.5
    df['prediction'] = (df['slider_value'] >= threshold).astype(int)
    df['true_label_int'] = df['true_label'].astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(df['true_label_int'], df['prediction'])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df['true_label_int'], df['prediction'], average='binary'
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(df['true_label_int'], df['prediction']).ravel()
    
    # Print results
    print(f"Human Evaluation Results ({len(df)} samples):")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_samples": len(df),
        "threshold": threshold
    }
    
    import json
    with open(os.path.join(output_dir, "human_eval_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot distribution of slider values
    plt.figure(figsize=(10, 6))
    plt.hist(df['slider_value'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Slider Value (0=Different Author, 1=Same Author)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Human Evaluation Scores')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "human_eval_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    """Main function for human evaluation CLI."""
    parser = argparse.ArgumentParser(description="Human Evaluation for Authorship Verification")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--analyze", "-a", help="Path to CSV file to analyze")
    parser.add_argument("--output-dir", "-o", default="results/human_evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    if args.config:
        run_human_evaluation(args.config)
    elif args.analyze:
        analyze_human_evaluation(args.analyze, args.output_dir)
    else:
        print("Please specify either --config to run evaluation or --analyze to analyze results")


if __name__ == "__main__":
    main()
