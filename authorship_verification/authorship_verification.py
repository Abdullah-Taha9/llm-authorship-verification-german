#!/usr/bin/env python3
"""
Authorship Verification Experiment Runner
Converted from preliminary_experiments.ipynb with YAML configuration support.
Updated with improved structure and configuration handling.
"""

import os
import re
import json
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Add data directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_loaders import get_data_loader


class AuthorshipVerificationExperiment:
    """Main class for running authorship verification experiments."""
    
    def __init__(self, config_path: str):
        """Initialize experiment with configuration."""
        self.config = self.load_config(config_path)
        self.client = self.setup_openai_client()
        self.prompts = self.setup_prompts()
        self.model_pricing = self.setup_model_pricing()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client with API key from .env file or api_key.txt."""
        api_key = None
        
        # Method 1: Try reading from api_key.txt file
        try:
            with open("api_key.txt", "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            print("✅ API key loaded from api_key.txt")
        except FileNotFoundError:
            pass
        
        # Method 2: Try loading from .env file
        if not api_key:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("✅ API key loaded from .env file")
        
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not found. Please either:\n"
                "1. Create api_key.txt file: echo 'your_api_key' > api_key.txt\n"
                "2. Create .env file with: OPENAI_API_KEY=your_api_key"
            )

        return OpenAI(api_key=api_key, base_url="https://llm.tensor.rocks/v1")
    
    def setup_prompts(self) -> Dict[str, str]:
        """Setup prompt templates based on configuration."""
        # English prompts
        lip_english = """
System instruction: Respond only with a single JSON object including three key elements: 
"analysis": Reasoning behind your answer in few words. 
"answer": A boolean (True/False) answer. 
"prob_same_author": A number between 0.0 and 1.0, where 0.0 = certain different author, 1.0 = certain same author, and values in between reflect degree of certainty (e.g., 0.9 = almost certain same, 0.1 = almost certain different). Ensure that "answer" matches: use True if prob_same_author ≥ 0.5, False if prob_same_author < 0.5.

Verify if two input texts were written by the same author. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reasoning based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings. 

Input text 1: 
\"\"\"{text1}\"\"\"

, text 2: 
\"\"\"{text2}\"\"\"

"""
        
        # German prompts  
        lip_german = """
System instruction: Antworte ausschließlich mit einem einzelnen JSON-Objekt, das drei Schlüssel enthält: 
"analysis": Begründung deiner Antwort in wenigen Worten. 
"answer": Eine boolesche Antwort (True/False). 
"prob_same_author": Eine Zahl zwischen 0.0 und 1.0, wobei 0.0 = sicher unterschiedlicher Autor, 1.0 = sicher gleicher Autor, und Werte dazwischen den Grad der Sicherheit widerspiegeln (z. B. 0.9 = fast sicher gleicher Autor, 0.1 = fast sicher unterschiedlicher Autor). Stelle sicher, dass "answer" dazu passt: True, wenn prob_same_author ≥ 0.5, False, wenn prob_same_author < 0.5.

Überprüfe, ob zwei Eingabetexte vom selben Autor geschrieben wurden. Analysiere die Schreibstile der Eingabetexte und ignoriere Unterschiede in Thema und Inhalt. Begründe deine Einschätzung anhand sprachlicher Merkmale wie Phrasalverben, Modalverben, Interpunktion, seltene Wörter, Affixe, Mengenangaben, Humor, Sarkasmus, Tippfehler und Rechtschreibfehler. 

Eingabetext 1: 
\"\"\"{text1}\"\"\"

, Text 2: 
\"\"\"{text2}\"\"\"

"""
        
        biased_english = """
System: You're a product‐review linguist on a shopping dataset. Texts are casual, everyday reviews.
Decide if two texts share an author, based purely on writing style.
Return JSON with {{analysis:Reasoning behind your answer, answer: A boolean (True/False)}}.

Keep your analysis **≤ 10 words**, **bullet-point** style (comma-separated is fine).

Example 1:
Text A: "They lacked any kind of moisture: wouldn't stay in my face they were so dry! Returning if possible!"
Text B: "Just as pictured. Prompt delivery. Yummmm"
analysis: "same casual contractions"
answer: true

Example 2:
Text A: "The box came really damaged. I got it for a gift and I am embarrassed to give it to him."
Text B: "Had to do some fabrication but it works. Not perfect or really flush but it gets the job done. You get what you pay for!"
analysis: "– emotive first-person  – technical, task-focused tone another person"
answer: false

Now your turn:
Text A:
\"\"\"{text1}\"\"\"

Text B:
\"\"\"{text2}\"\"\"
"""
        
        # Map prompts based on configuration
        prompt_map = {
            ("lip", "english"): lip_english,
            ("lip", "german"): lip_german,
            ("biased", "english"): biased_english,
            ("biased", "german"): biased_english  # Using English version for now
        }
        
        prompt_key = (self.config["prompt"], self.config["prompt_language"])
        if prompt_key not in prompt_map:
            raise ValueError(f"Unsupported prompt configuration: {prompt_key}")
            
        return {f"{self.config['prompt']}_{self.config['prompt_language']}": prompt_map[prompt_key]}
    
    def setup_model_pricing(self) -> Dict[str, tuple]:
        """Setup model pricing information."""
        return {
            "deepseek/deepseek-r1": (1.35, 5.40),
            "openai/gpt-4.1": (2.00, 8.00),
            "openai/gpt-4o": (2.50, 10.00),
            "openai/gpt-4o-mini": (0.17, 0.66),
            "openai/gpt-4.1-mini": (0.40, 1.60),
            "deepseek/deepseek-v3": (1.14, 4.56),
            "openai/o3-mini": (1.10, 4.40),
            "openai/gpt-4.1-nano": (0.10, 0.40),
        }
    
    def load_dataset(self):
        """Load dataset based on configuration using data_loaders module."""
        return get_data_loader(self.config)
    

    
    def extract_json(self, text: str) -> str:
        """Extract JSON from model response."""
        # Remove markdown code block fences
        text = re.sub(r"^```(json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.IGNORECASE)
        
        # Find first {...}
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text
    
    def save_partial(self, path: str, data: list) -> None:
        """Save partial or final results to a JSONl file."""
        with open(path, "w", encoding="utf-8") as f_out:
            for row in data:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    def generate_output_paths(self) -> Dict[str, str]:
        """Generate output file paths using configuration structure."""
        model_short = self.config["model"].split('/')[-1]
        language = self.config["dataset_language"]
        prompt = self.config["prompt"]
        data_loader = self.config.get("data_loader")
        
        # Create clean directory name: {language}_{prompt}_{data_loader}_{model}
        config_parts = [language, prompt]
        if data_loader:
            config_parts.append(data_loader)
        config_parts.append(model_short)
        
        config_name = "_".join(config_parts)
        output_dir = Path(self.config["output_dir"]) / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base filename follows same pattern
        base_name = config_name
        
        return {
            "jsonl": output_dir / f"{base_name}.jsonl",
            "metrics": output_dir / f"{base_name}.metrics.json",
            "costs": output_dir / f"{base_name}.costs.json",
            "csv": output_dir / f"{base_name}.csv"
        }
    
    def run_experiment(self):
        """Run the main experiment."""
        print(f"Starting experiment with configuration:")
        print(f"  Dataset: {self.config['dataset']} ({self.config['dataset_language']})")
        print(f"  Prompt: {self.config['prompt']} ({self.config['prompt_language']})")
        print(f"  Model: {self.config['model']}")
        print(f"  Max samples: {self.config['max_samples']}")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Get model and pricing info
        model = self.config["model"]
        model_short = model.split('/')[-1]
        input_cost_per_million, output_cost_per_million = self.model_pricing[model]
        
        # Get prompt
        prompt_key = f"{self.config['prompt']}_{self.config['prompt_language']}"
        prompt_template = self.prompts[prompt_key]
        
        # Generate output paths
        paths = self.generate_output_paths()
        
        print(f"\nEvaluating {model_short} on {len(dataset)} samples...")
        
        results = []
        correct = 0
        
        for idx, item in enumerate(tqdm(dataset, desc=f"{model_short}-experiment")):
            text1 = item["review_1"]["review_body"]
            text2 = item["review_2"]["review_body"]
            target = item["label"]
            
            prompt = prompt_template.format(text1=text1, text2=text2)
            chat_messages = [{"role": "user", "content": prompt}]
            orig_idx = item.get("orig_idx")
            
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content.strip()
                content_json = self.extract_json(content)
                
                # Extract token usage
                if getattr(response, "usage", None):
                    prompt_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                else:
                    prompt_tokens = output_tokens = total_tokens = None
                
                # Parse JSON response
                parsed = json.loads(content_json)
                answer = bool(parsed.get("answer"))
                analysis = parsed.get("analysis", "")
                
                try:
                    prob_same_author = float(parsed.get("prob_same_author"))
                except (TypeError, ValueError):
                    prob_same_author = None
                    
            except json.JSONDecodeError as exc:
                answer = None
                analysis = f"JSON Parse Error: {exc}. Raw response: {content}..."
                parsed = {}
                prob_same_author = None
                prompt_tokens = output_tokens = total_tokens = None
                
            except Exception as exc:
                answer = None
                analysis = f"General Error: {exc}"
                parsed = {}
                prob_same_author = None
                prompt_tokens = output_tokens = total_tokens = None
            
            is_correct = answer is not None and answer == target
            correct += int(is_correct)
            
            results.append({
                "idx": idx,
                "orig_idx": orig_idx,
                "target": target,
                "LLM_answer": answer,
                "prob_same_author": prob_same_author,
                "correct_response": is_correct,
                "analysis": analysis,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "prompt_used": prompt,
                "text1": text1,
                "text2": text2,
                "raw_response": content if 'content' in locals() else None
            })
            
            # Periodically save partial results
            if (idx + 1) % 100 == 0:
                self.save_partial(str(paths["jsonl"]), results)
        
        # Calculate metrics based on configuration
        metrics_config = self.config.get("metrics", {})
        calculated_metrics = {}
        
        # Get valid predictions for metric calculation
        valid_results = [r for r in results if r.get("LLM_answer") is not None]
        if not valid_results:
            print("Warning: No valid predictions found!")
            return {}
            
        true_labels = [r["target"] for r in valid_results]
        pred_labels = [r["LLM_answer"] for r in valid_results]
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[False, True]).ravel()
        
        # Calculate metrics based on configuration
        if metrics_config.get("accuracy", True):
            calculated_metrics["accuracy"] = correct / len(results) if len(results) > 0 else 0
            
        if metrics_config.get("precision", True):
            calculated_metrics["precision"] = precision_score(true_labels, pred_labels, zero_division=0)
            
        if metrics_config.get("recall", True):
            calculated_metrics["recall"] = recall_score(true_labels, pred_labels, zero_division=0)
            
        if metrics_config.get("f1_score", True):
            calculated_metrics["f1_score"] = f1_score(true_labels, pred_labels, zero_division=0)
            
        if metrics_config.get("specificity", True):
            calculated_metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        if metrics_config.get("npv", True):  # Negative Predictive Value
            calculated_metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
        # Confusion matrix components
        if metrics_config.get("tp", True):
            calculated_metrics["true_positive"] = int(tp)
        if metrics_config.get("tn", True):
            calculated_metrics["true_negative"] = int(tn)
        if metrics_config.get("fp", True):
            calculated_metrics["false_positive"] = int(fp)
        if metrics_config.get("fn", True):
            calculated_metrics["false_negative"] = int(fn)
            
        # Token usage metrics
        if metrics_config.get("token_usage", True):
            calculated_metrics["total_input_tokens"] = sum(r.get("prompt_tokens", 0) for r in results if r.get("prompt_tokens") is not None)
            calculated_metrics["total_output_tokens"] = sum(r.get("output_tokens", 0) for r in results if r.get("output_tokens") is not None)
            calculated_metrics["total_tokens_used"] = sum(r.get("total_tokens", 0) for r in results if r.get("total_tokens") is not None)
        
        # Calculate costs
        total_input_tokens = calculated_metrics.get("total_input_tokens", 0)
        total_output_tokens = calculated_metrics.get("total_output_tokens", 0)
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost
        
        # Save results
        if self.config.get("save_responses", True):
            self.save_partial(str(paths["jsonl"]), results)
        
        if self.config.get("save_metrics", True):
            # Base experiment info
            metrics = {
                "experiment_name": self.config.get("experiment_name", "experiment"),
                "model": model_short,
                "language": self.config["dataset_language"],
                "prompt": self.config["prompt"],
                "data_loader": self.config.get("data_loader"),
                "total": len(results),
                "correct": correct,
            }
            
            # Add calculated metrics
            metrics.update(calculated_metrics)
            
            with open(paths["metrics"], "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        if self.config.get("save_costs", True) and metrics_config.get("costs", True):
            costs = {
                "experiment_name": self.config.get("experiment_name", "experiment"),
                "model": model_short,
                "language": self.config["dataset_language"],
                "prompt": self.config["prompt"],
                "input_cost_per_million": input_cost_per_million,
                "output_cost_per_million": output_cost_per_million,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost_usd": total_cost,
            }
            
            with open(paths["costs"], "w", encoding="utf-8") as f:
                json.dump(costs, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        if self.config.get("save_csv", True):
            df = pd.DataFrame(results)
            df.to_csv(paths["csv"], index=False)
        
        print(f"\nExperiment completed!")
        print(f"Accuracy: {calculated_metrics.get('accuracy', 'N/A'):.3f}")
        print(f"F1-Score: {calculated_metrics.get('f1_score', 'N/A'):.3f}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Results saved to: {paths['jsonl'].parent}")
        
        # Return summary
        summary = {
            "experiment_name": self.config.get("experiment_name", "experiment"),
            "model": model_short,
            "language": self.config["dataset_language"],
            "prompt": self.config["prompt"],
            "total": len(results),
            "correct": correct,
            "cost": total_cost
        }
        summary.update(calculated_metrics)
        
        return summary


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Run Authorship Verification Experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Run experiment
    experiment = AuthorshipVerificationExperiment(args.config)
    results = experiment.run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Language: {results['language']}")
    print(f"Prompt: {results['prompt']}")
    print(f"Total Samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    print(f"Cost: ${results['cost']:.4f}")


if __name__ == "__main__":
    main()
