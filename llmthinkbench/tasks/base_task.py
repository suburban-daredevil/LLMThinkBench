# Updated base_task.py with proper token counting

from abc import ABC, abstractmethod
import logging
import os
import json
import numpy as np
import time
import traceback
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from google import genai

# Add these imports for proper token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

class BaseTask(ABC):
    """Abstract base class for all evaluation tasks with enhanced error handling and retry mechanisms"""
    
    def __init__(self, model_handler, output_dir, min_val, max_val, num_folds, 
                 num_samples, store_details, temperature, top_p, max_tokens, seed=None):
        """
        Initialize base task with common parameters
        
        Args:
            model_handler: Model handler for inference (supports both vLLM and Transformers)
            output_dir: Directory to save results
            min_val: Minimum value for number generation
            max_val: Maximum value for number generation
            num_folds: Number of evaluation folds
            num_samples: Number of samples to generate per test case
            store_details: Whether to store detailed results
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
        """
        self.model_handler = model_handler
        self.output_dir = output_dir
        self.min_val = min_val
        self.max_val = max_val
        self.num_folds = num_folds
        self.num_samples = num_samples
        self.store_details = store_details
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        
        # Retry configuration
        self.max_retries = 3  # Maximum number of retries per sample
        self.retry_delay = 1.0  # Initial delay between retries (seconds)
        self.max_consecutive_failures = 10  # Maximum consecutive failures before stopping
        
        # Initialize token counter
        self._init_token_counter()
        
        # Create task-specific directory
        self.task_dir = os.path.join(output_dir, self.task_name)
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Log model information
        model_info = self.model_handler.get_model_info()
        logging.info(f"🤖 Model Configuration for {self.task_name}:")
        for key, value in model_info.items():
            logging.info(f"   - {key}: {value}")
        
        # Log token counting method
        logging.info(f"🔢 Token counting method: {self.token_counting_method}")
    
    def _init_token_counter(self):
        """Initialize the appropriate token counter based on available libraries and model"""
        self.token_counter = None
        self.token_counting_method = "word_estimate"
        
        try:
            # Get model info to determine the best tokenizer
            model_info = self.model_handler.get_model_info()
            model_name = model_info.get('model_name', '').lower()
            
            # Try to initialize tiktoken for OpenAI models or as a general fallback
            if TIKTOKEN_AVAILABLE:
                try:
                    # Map common model names to tiktoken encodings
                    if any(name in model_name for name in ['gpt-4', 'gpt-3.5', 'text-davinci', 'text-curie']):
                        if 'gpt-4' in model_name:
                            self.token_counter = tiktoken.encoding_for_model("gpt-4")
                        elif 'gpt-3.5' in model_name:
                            self.token_counter = tiktoken.encoding_for_model("gpt-3.5-turbo")
                        else:
                            self.token_counter = tiktoken.encoding_for_model("text-davinci-003")
                        self.token_counting_method = f"tiktoken_{model_name}"
                    else:
                        # Use cl100k_base as a general-purpose encoder for most modern models
                        self.token_counter = tiktoken.get_encoding("cl100k_base")
                        self.token_counting_method = "tiktoken_cl100k_base"
                    
                    logging.info(f"✅ Initialized tiktoken encoder: {self.token_counting_method}")
                    return
                    
                except Exception as e:
                    logging.warning(f"⚠️  Failed to initialize tiktoken: {e}")
            
            # Try to use the model's own tokenizer if available
            if hasattr(self.model_handler, 'tokenizer') and self.model_handler.tokenizer is not None:
                try:
                    self.token_counter = self.model_handler.tokenizer
                    self.token_counting_method = "model_tokenizer"
                    logging.info("✅ Using model's own tokenizer for token counting")
                    return
                except Exception as e:
                    logging.warning(f"⚠️  Failed to use model tokenizer: {e}")
            
            # Try to load tokenizer from transformers library
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Try to load tokenizer based on model name
                    if model_name:
                        # Common model name mappings
                        tokenizer_name = model_name
                        if 'llama' in model_name:
                            tokenizer_name = "meta-llama/Llama-2-7b-hf"  # Default Llama tokenizer
                        elif 'mistral' in model_name:
                            tokenizer_name = "mistralai/Mistral-7B-v0.1"
                        elif 'falcon' in model_name:
                            tokenizer_name = "tiiuae/falcon-7b"
                        
                        self.token_counter = AutoTokenizer.from_pretrained(tokenizer_name)
                        self.token_counting_method = f"transformers_{tokenizer_name}"
                        logging.info(f"✅ Loaded transformers tokenizer: {tokenizer_name}")
                        return
                        
                except Exception as e:
                    logging.warning(f"⚠️  Failed to load transformers tokenizer: {e}")
            
            # Fallback to word-based estimation
            logging.warning("⚠️  Using word-based token estimation. Install tiktoken for accurate counting.")
            
        except Exception as e:
            logging.error(f"❌ Error initializing token counter: {e}")
            logging.warning("⚠️  Falling back to word-based estimation")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the best available method
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            if self.token_counter is not None:
                if self.token_counting_method.startswith("tiktoken"):
                    # Using tiktoken
                    return len(self.token_counter.encode(text))
                
                elif self.token_counting_method == "model_tokenizer":
                    # Using model's tokenizer
                    if hasattr(self.token_counter, 'encode'):
                        tokens = self.token_counter.encode(text)
                        return len(tokens) if isinstance(tokens, list) else tokens.size(0)
                    elif hasattr(self.token_counter, 'tokenize'):
                        return len(self.token_counter.tokenize(text))
                
                elif self.token_counting_method.startswith("transformers"):
                    # Using transformers tokenizer
                    tokens = self.token_counter.encode(text, add_special_tokens=False)
                    return len(tokens)
            
            # Fallback to improved word-based estimation
            return self._estimate_tokens_from_words(text)
            
        except Exception as e:
            logging.warning(f"⚠️  Token counting failed: {e}, using word estimation")
            return self._estimate_tokens_from_words(text)
    
    def _estimate_tokens_from_words(self, text: str) -> int:
        """
        Improved word-based token estimation
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        # Split by whitespace and punctuation for better estimation
        import re
        
        # Count words and punctuation separately
        words = len(re.findall(r'\b\w+\b', text))
        punctuation = len(re.findall(r'[^\w\s]', text))
        
        # Improved estimation based on empirical observations:
        # - Most words are 1-2 tokens
        # - Punctuation is usually 1 token
        # - Account for subword tokenization
        estimated_tokens = int(words * 1.3 + punctuation * 0.8)
        
        return max(1, estimated_tokens)  # Ensure at least 1 token for non-empty text
    
    @property
    @abstractmethod
    def task_name(self):
        """Return task name, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def generate_data(self, **kwargs):
        """Generate evaluation data, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def create_prompt(self, data_point):
        """Create prompt for the task, to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def evaluate_response(self, response, data_point):
        """Evaluate model response, to be implemented by subclasses"""
        pass
    
    def save_detailed_results(self, results, test_case_id, fold):
        """Save detailed results for each test case and fold"""
        if not self.store_details:
            return
            
        case_dir = os.path.join(self.task_dir, f"test_case_{test_case_id}")
        os.makedirs(case_dir, exist_ok=True)
        
        filename = os.path.join(case_dir, f"detailed_results_fold_{fold}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
            logging.info(f"💾 Saved detailed results for test case {test_case_id} fold {fold} to {filename}")
        except Exception as e:
            logging.error(f"❌ Failed to save detailed results: {e}")
    
    def process_fold_metrics(self, fold_results):
        """Calculate metrics for a single fold"""
        if not fold_results:
            logging.warning("⚠️  No results to process for fold metrics")
            return {
                'total': 0,
                'correct': 0,
                'instruction_followed': 0,
                'accuracy': 0.0,
                'instruction_followed_pct': 0.0,
                'avg_response_length': 0.0,
                'avg_word_count': 0.0,
                'avg_output_tokens': 0.0,
                'generation_failures': 0,
                'parsing_failures': 0
            }
        
        # Count different types of results
        successful_results = [r for r in fold_results if r.get('generation_success', True)]
        generation_failures = len(fold_results) - len(successful_results)
        parsing_failures = sum(1 for r in successful_results if not r.get('parsing_success', True))
        
        # Calculate metrics only on successful results
        if successful_results:
            metrics = {
                'total': len(fold_results),
                'successful_generations': len(successful_results),
                'correct': sum(item.get('accuracy', 0) for item in successful_results),
                'instruction_followed': sum(item.get('instruction_followed', 0) for item in successful_results),
                'response_lengths': [item.get('string_len', 0) for item in successful_results],
                'word_counts': [item.get('words', 0) for item in successful_results],
                'output_tokens': [item.get('tokens', 0) for item in successful_results],
                'generation_failures': generation_failures,
                'parsing_failures': parsing_failures
            }
            
            # Calculate percentages and averages
            metrics['accuracy'] = round(metrics['correct'] / len(successful_results), 4) if successful_results else 0.0
            metrics['instruction_followed_pct'] = round(metrics['instruction_followed'] / len(successful_results), 4) if successful_results else 0.0
            metrics['avg_response_length'] = round(np.mean(metrics['response_lengths']), 2) if metrics['response_lengths'] else 0.0
            metrics['avg_word_count'] = round(np.mean(metrics['word_counts']), 2) if metrics['word_counts'] else 0.0
            metrics['avg_output_tokens'] = round(np.mean(metrics['output_tokens']), 2) if metrics['output_tokens'] else 0.0
            
            # Calculate success rates
            metrics['generation_success_rate'] = round(len(successful_results) / len(fold_results), 4)
            metrics['parsing_success_rate'] = round((len(successful_results) - parsing_failures) / len(successful_results), 4) if successful_results else 0.0
            
        else:
            # All generations failed
            metrics = {
                'total': len(fold_results),
                'successful_generations': 0,
                'correct': 0,
                'instruction_followed': 0,
                'accuracy': 0.0,
                'instruction_followed_pct': 0.0,
                'avg_response_length': 0.0,
                'avg_word_count': 0.0,
                'avg_output_tokens': 0.0,
                'generation_failures': generation_failures,
                'parsing_failures': 0,
                'generation_success_rate': 0.0,
                'parsing_success_rate': 0.0
            }
        
        return metrics
    
    def generate_response_with_retry(self, prompt: str, data_point: Dict) -> Tuple[str, int, bool, Dict]:
        """
        Generate response with retry mechanism for failed generations
        
        Args:
            prompt: The input prompt
            data_point: The data point being processed
            
        Returns:
            Tuple of (response, tokens, success, metadata)
        """
        last_error = None
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                # Determine backend and generate accordingly
                if self.model_handler.backend == 'vllm':
                    response, tokens = self._generate_vllm_response(prompt)
                elif self.model_handler.backend == 'transformers':
                    response, tokens = self._generate_transformers_response(prompt)
                else:
                    raise RuntimeError(f"Unknown backend: {self.model_handler.backend}")
                
                # If we get here, generation was successful
                metadata = {
                    'attempt': attempt + 1,
                    'backend': self.model_handler.backend,
                    'retry_count': retry_count,
                    'token_counting_method': self.token_counting_method
                }
                
                if attempt > 0:
                    logging.info(f"✅ Generation successful on attempt {attempt + 1}")
                
                return response, tokens, True, metadata
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                error_msg = str(e)
                logging.warning(f"⚠️  Generation attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < self.max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    logging.info(f"🔄 Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
                    # For some errors, try regenerating the data point
                    if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                        logging.info("🔄 CUDA/Memory error detected, regenerating data point...")
                        try:
                            # Attempt to regenerate data point (implementation specific)
                            new_data_point = self.regenerate_data_point(data_point)
                            if new_data_point != data_point:
                                logging.info("✨ Generated new data point for retry")
                                data_point.update(new_data_point)
                                prompt = self.create_prompt(data_point)
                        except Exception as regen_error:
                            logging.warning(f"⚠️  Failed to regenerate data point: {regen_error}")
        
        # All retries failed
        logging.error(f"❌ All {self.max_retries + 1} generation attempts failed. Last error: {last_error}")
        
        metadata = {
            'attempt': self.max_retries + 1,
            'backend': self.model_handler.backend,
            'retry_count': retry_count,
            'final_error': str(last_error),
            'token_counting_method': self.token_counting_method
        }
        
        return "", 0, False, metadata
    
    def _generate_vllm_response(self, prompt: str) -> Tuple[str, int]:
        """Generate response using vLLM backend with accurate token counting"""
        # For vLLM, we need to format the prompt according to the chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.model_handler.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Use the model_handler's generate method
        responses = self.model_handler.generate(
            prompts=[formatted_prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        response = responses[0] if responses else ""
        
        # Count tokens accurately
        tokens = self.count_tokens(response)
        
        return response, tokens
    
    def _generate_transformers_response(self, prompt: str) -> Tuple[str, int]:
        """Generate response using Transformers backend with accurate token counting"""
        # Use the model_handler's generate method
        responses = self.model_handler.generate(
            prompts=[prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        response = responses[0] if responses else ""
        
        # Count tokens accurately
        tokens = self.count_tokens(response)
        
        return response, tokens
    
    def regenerate_data_point(self, original_data_point: Dict) -> Dict:
        """
        Regenerate a data point when the original fails
        Default implementation - subclasses can override for specific logic
        
        Args:
            original_data_point: The original data point that failed
            
        Returns:
            New data point dictionary
        """
        try:
            # Generate a new data point using the same parameters
            new_data = self.generate_data()
            if isinstance(new_data, list) and len(new_data) > 0:
                return new_data[0]  # Return first item if it's a list
            elif isinstance(new_data, dict):
                return new_data
            else:
                logging.warning("⚠️  Regenerated data is not in expected format")
                return original_data_point
        except Exception as e:
            logging.warning(f"⚠️  Failed to regenerate data point: {e}")
            return original_data_point
        
    # Function to get response from Gemini model
    def generate_response_with_gemini(self, prompt, data_point):
        
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL")

        client = genai.Client(api_key=api_key)

        try:
            # Generate response using gemini model
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            # Return the response object
            return response, response.usage.total_tokens, True, {
                'model': model_name,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'data_point_id': data_point.get('id', 'unknown')
            }
        except Exception as e:
            print(f"Error getting response for prompt: {e}")
            return f"Error: {str(e)}"

        
    def run_fold(self, data, test_case_id, fold):
        """Run a single evaluation fold with enhanced error handling"""
        fold_results = []
        consecutive_failures = 0
        
        progress_bar = tqdm(data, desc=f"Test case {test_case_id} - Fold {fold}")
        
        for i, data_point in enumerate(progress_bar):
            try:
                # Create prompt
                prompt = self.create_prompt(data_point)
                
                # Generate response with retry mechanism
                # response, tokens, generation_success, generation_metadata = self.generate_response_with_retry(
                #     prompt, data_point
                # )

                response, tokens, generation_success, generation_metadata = self.generate_response_with_gemini(
                    prompt, data_point
                )
                
                # Initialize result with generation info
                eval_result = {
                    "prompt": prompt,
                    "model_response": response,
                    "string_len": len(response),
                    "words": len(response.split()),
                    "tokens": tokens,  # Now using accurate token counting
                    "generation_success": generation_success,
                    "generation_metadata": generation_metadata
                }
                
                if generation_success:
                    # Reset consecutive failures counter
                    consecutive_failures = 0
                    
                    # Evaluate response
                    try:
                        response_eval = self.evaluate_response(response, data_point)
                        eval_result.update(response_eval)
                        eval_result["parsing_success"] = True
                        
                    except Exception as eval_error:
                        logging.warning(f"⚠️  Response evaluation failed for sample {i}: {eval_error}")
                        eval_result.update({
                            "accuracy": 0,
                            "instruction_followed": 0,
                            "parsing_success": False,
                            "parsing_error": str(eval_error),
                            "predicted_answer": None,
                            "ground_truth": data_point["answer"] if data_point is not None else "unknown"
                        })
                
                else:
                    # Generation failed completely
                    consecutive_failures += 1
                    eval_result.update({
                        "accuracy": 0,
                        "instruction_followed": 0,
                        "parsing_success": False,
                        "parsing_error": "Generation failed",
                        "predicted_answer": None,
                        "ground_truth": data_point.get("answer", "unknown")
                    })
                    
                    # Check if we should stop due to too many consecutive failures
                    if consecutive_failures >= self.max_consecutive_failures:
                        logging.error(f"❌ Stopping evaluation due to {consecutive_failures} consecutive failures")
                        break
                
                fold_results.append(eval_result)
                
                # Update progress bar with current stats
                if len(fold_results) > 0:
                    current_accuracy = sum(r.get('accuracy', 0) for r in fold_results) / len(fold_results)
                    progress_bar.set_postfix({
                        'acc': f"{current_accuracy:.3f}",
                        'fails': consecutive_failures
                    })
                
            except KeyboardInterrupt:
                logging.info("⏹️  Evaluation interrupted by user")
                break
                
            except Exception as e:
                logging.error(f"❌ Unexpected error processing sample {i}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                
                # Add a failed result
                fold_results.append({
                    "prompt": "Error occurred before prompt creation",
                    "model_response": "",
                    "string_len": 0,
                    "words": 0,
                    "tokens": 0,
                    "generation_success": False,
                    "parsing_success": False,
                    "accuracy": 0,
                    "instruction_followed": 0,
                    "error": str(e),
                    "predicted_answer": None,
                    "ground_truth": data_point.get("answer", "unknown")
                })
                
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    logging.error(f"❌ Stopping evaluation due to {consecutive_failures} consecutive failures")
                    break
        
        progress_bar.close()
        
        # Save detailed results if requested
        self.save_detailed_results(fold_results, test_case_id, fold)
        
        # Calculate fold metrics
        metrics = self.process_fold_metrics(fold_results)
        metrics.update({
            'test_case_id': test_case_id,
            'fold': fold,
            'task': self.task_name,
            'total_samples_attempted': len(data),
            'total_samples_completed': len(fold_results),
            'token_counting_method': self.token_counting_method
        })
        
        # Log fold summary
        logging.info(f"📊 Fold {fold} Summary:")
        logging.info(f"   - Samples: {metrics['total_samples_completed']}/{metrics['total_samples_attempted']}")
        logging.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"   - Generation Success Rate: {metrics.get('generation_success_rate', 0):.4f}")
        logging.info(f"   - Parsing Success Rate: {metrics.get('parsing_success_rate', 0):.4f}")
        logging.info(f"   - Avg Tokens per Response: {metrics.get('avg_output_tokens', 0):.1f}")
        
        return metrics
    
    def run_evaluation(self, list_sizes: Optional[List[int]] = None) -> List[Dict]:
        """
        Run complete evaluation across all test cases and folds
        
        Args:
            list_sizes: List of sizes to test (for applicable tasks)
            
        Returns:
            List of metrics dictionaries
        """
        all_metrics = []

        sleep_seconds = int(os.getenv("SECONDS_TIME_OUT_BETWEEN_REQUESTS"))
        
        try:
            # Determine test cases based on task type
            if list_sizes is not None:
                test_cases = list_sizes
                logging.info(f"🎯 Running {self.task_name} evaluation with list sizes: {test_cases}")
            else:
                test_cases = [None]  # Single test case for non-list tasks
                logging.info(f"🎯 Running {self.task_name} evaluation")
            
            for test_case_id, test_case in enumerate(test_cases):
                logging.info(f"\n{'='*50}")
                logging.info(f"📋 Test Case {test_case_id + 1}: {test_case if test_case is not None else 'Default'}")
                logging.info(f"{'='*50}")
                
                # Generate data for this test case
                try:
                    if test_case is not None:
                        data = self.generate_data(list_size=test_case)
                    else:
                        data = self.generate_data()
                    
                    logging.info(f"✅ Generated {len(data)} samples for test case")
                    
                except Exception as e:
                    logging.error(f"❌ Failed to generate data for test case {test_case}: {e}")
                    continue
                
                # Run all folds for this test case
                for fold in range(self.num_folds):
                    try:
                        fold_metrics = self.run_fold(data, test_case_id, fold)

                        time.sleep(sleep_seconds)  # Respect rate limits
                        logging.info(f"Sleeping for {sleep_seconds} seconds for rate limit compliance...")

                        all_metrics.append(fold_metrics)
                        
                    except Exception as e:
                        logging.error(f"❌ Failed to complete fold {fold} for test case {test_case}: {e}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        continue
            
            # Log overall summary
            if all_metrics:
                avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
                avg_gen_success = np.mean([m.get('generation_success_rate', 0) for m in all_metrics])
                avg_tokens = np.mean([m.get('avg_output_tokens', 0) for m in all_metrics])
                logging.info(f"\n🎉 {self.task_name} Evaluation Complete!")
                logging.info(f"📊 Overall Results:")
                logging.info(f"   - Average Accuracy: {avg_accuracy:.4f}")
                logging.info(f"   - Average Generation Success Rate: {avg_gen_success:.4f}")
                logging.info(f"   - Average Tokens per Response: {avg_tokens:.1f}")
                logging.info(f"   - Token Counting Method: {self.token_counting_method}")
                logging.info(f"   - Total Metrics Collected: {len(all_metrics)}")
            else:
                logging.warning(f"⚠️  No metrics collected for {self.task_name}")
                
        except Exception as e:
            logging.error(f"❌ Critical error in {self.task_name} evaluation: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        
        return all_metrics