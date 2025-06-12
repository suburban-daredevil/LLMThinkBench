# model_handler.py
import logging
import torch
from typing import Optional, List, Dict, Any

# Model loading and inference handler with fallback support
class ModelHandler:
    """
    Enhanced model handler with vLLM primary and Transformers fallback support.
    
    This handler attempts to load models using vLLM for optimal performance,
    and automatically falls back to Transformers if vLLM fails or doesn't support the model.
    """
    
    def __init__(self, model_id: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9, trust_remote_code: bool = False):
        """
        Initialize model handler with automatic fallback mechanism.
        
        Args:
            model_id: Hugging Face model identifier
            tensor_parallel_size: Number of GPUs to use (vLLM only)
            gpu_memory_utilization: GPU memory utilization threshold (vLLM only)
            trust_remote_code: Whether to trust remote code execution (Transformers only)
        """
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.backend = None  # Will be set to 'vllm' or 'transformers'
        self.model = None
        self.tokenizer = None
        
        # Attempt to load model with fallback
        self._load_model_with_fallback()
    
    def _load_model_with_fallback(self):
        """
        Load model with vLLM first, fallback to Transformers if needed.
        """
        # First attempt: vLLM
        if self._try_load_vllm():
            logging.info(f"✅ Successfully loaded model '{self.model_id}' using vLLM backend")
            self.backend = 'vllm'
            return
        
        # Fallback: Transformers
        if self._try_load_transformers():
            logging.info(f"✅ Successfully loaded model '{self.model_id}' using Transformers backend (fallback)")
            self.backend = 'transformers'
            return
        
        # Both failed
        error_msg = f"❌ Failed to load model '{self.model_id}' with both vLLM and Transformers backends"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _try_load_vllm(self) -> bool:
        """
        Attempt to load model using vLLM.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"🔄 Attempting to load model '{self.model_id}' with vLLM...")
            
            from vllm import LLM
            from transformers import AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model with vLLM
            self.model = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=8192,
                gpu_memory_utilization=self.gpu_memory_utilization,
                # Note: vLLM doesn't have trust_remote_code parameter in the same way
            )
            
            logging.info(f"📊 vLLM configuration:")
            logging.info(f"   - Model: {self.model_id}")
            logging.info(f"   - Tensor parallel size: {self.tensor_parallel_size}")
            logging.info(f"   - GPU memory utilization: {self.gpu_memory_utilization}")
            logging.info(f"   - Max model length: 8192")
            
            return True
            
        except ImportError as e:
            logging.warning(f"⚠️  vLLM not available: {e}")
            return False
        except Exception as e:
            logging.warning(f"⚠️  Failed to load model with vLLM: {e}")
            return False
    
    def _try_load_transformers(self) -> bool:
        """
        Attempt to load model using Transformers.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"🔄 Attempting to load model '{self.model_id}' with Transformers...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=self.trust_remote_code
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
            if device == "cpu":
                logging.warning("⚠️  No CUDA available, using CPU. Performance will be significantly slower.")
            
            logging.info(f"📊 Transformers configuration:")
            logging.info(f"   - Model: {self.model_id}")
            logging.info(f"   - Device: {device}")
            logging.info(f"   - Trust remote code: {self.trust_remote_code}")
            logging.info(f"   - Data type: {'float16' if device == 'cuda' else 'float32'}")
            
            return True
            
        except ImportError as e:
            logging.error(f"❌ Transformers not available: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to load model with Transformers: {e}")
            return False
    
    def generate(self, prompts: List[str], temperature: float = 0.7, 
                 top_p: float = 0.9, max_tokens: int = 512) -> List[str]:
        """
        Generate responses using the loaded model.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated responses
        """
        if self.backend == 'vllm':
            return self._generate_vllm(prompts, temperature, top_p, max_tokens)
        elif self.backend == 'transformers':
            return self._generate_transformers(prompts, temperature, top_p, max_tokens)
        else:
            raise RuntimeError("No backend available for generation")
    
    def _generate_vllm(self, prompts: List[str], temperature: float, 
                       top_p: float, max_tokens: int) -> List[str]:
        """Generate using vLLM backend."""
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)
            return [output.outputs[0].text for output in outputs]
            
        except Exception as e:
            logging.error(f"❌ Error during vLLM generation: {e}")
            raise e
    
    def _generate_transformers(self, prompts: List[str], temperature: float, 
                              top_p: float, max_tokens: int) -> List[str]:
        """Generate using Transformers backend."""
        try:
            results = []
            
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True
                )
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode response (exclude input tokens)
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(response)
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Error during Transformers generation: {e}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_id': self.model_id,
            'backend': self.backend,
            'tensor_parallel_size': self.tensor_parallel_size if self.backend == 'vllm' else 1,
            'gpu_memory_utilization': self.gpu_memory_utilization if self.backend == 'vllm' else 'auto',
            'trust_remote_code': self.trust_remote_code,
        }
        
        if self.backend == 'transformers':
            info['device'] = str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
            info['dtype'] = str(self.model.dtype)
        
        return info
    
    def __str__(self) -> str:
        """String representation of the model handler."""
        return f"ModelHandler(model_id='{self.model_id}', backend='{self.backend}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model handler."""
        return (f"ModelHandler(model_id='{self.model_id}', backend='{self.backend}', "
                f"tensor_parallel_size={self.tensor_parallel_size}, "
                f"gpu_memory_utilization={self.gpu_memory_utilization}, "
                f"trust_remote_code={self.trust_remote_code})")