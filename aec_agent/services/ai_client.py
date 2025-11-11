"""AI client integration (OpenAI/Gemini compatible)."""

from typing import Dict, Any, Optional
import logging

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.config import config
from ..core.logger import get_logger


class AIClient:
    """
    OpenAI client for compliance analysis.
    
    Handles API calls to OpenAI models for building compliance verification.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        self.logger = get_logger(__name__)
        self.client = None
        self.api_key = api_key or config.openai_api_key
        self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided. AI integration will not work.")
            return
        
        if not OPENAI_AVAILABLE:
            self.logger.warning("openai package not available. Install with: pip install openai")
            return
        
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.logger.info("Initialized OpenAI client")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def analyze_compliance(self, prompt: str, model_name: str = None) -> Dict[str, Any]:
        """
        Analyze compliance using OpenAI.
        
        Args:
            prompt: Analysis prompt
            model_name: Optional model name override
            
        Returns:
            Analysis results
        """
        if not self.api_key:
            return {
                "error": "No OpenAI API key configured",
                "status": "failed"
            }
        
        if not OPENAI_AVAILABLE:
            return {
                "error": "openai package not available",
                "status": "failed",
                "solution": "Install with: pip install openai"
            }
        
        if not self.client:
            return {
                "error": "OpenAI client not initialized",
                "status": "failed"
            }
        
        try:
            model = model_name or config.default_model
            if not model.startswith(('gpt-', 'o1-')):
                model = "gpt-4"  # Default to GPT-4 if not a valid OpenAI model
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "status": "success",
                "analysis": analysis_text,
                "model_used": model,
                "prompt_length": len(prompt),
                "response_length": len(analysis_text) if analysis_text else 0,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return {
                "error": f"OpenAI API call failed: {e}",
                "status": "failed"
            }
    
    def check_connection(self) -> bool:
        """Check if connection to OpenAI API is working."""
        if not self.api_key or not OPENAI_AVAILABLE or not self.client:
            return False
        
        try:
            # Test with simple completion
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            return False