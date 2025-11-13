"""
LLM Guardrails - Simple external guardrails for LLM calls and reactive execution.

This module provides basic external safeguards to prevent runaway LLM calls,
infinite loops, and memory bloat through simple caps and limits.
"""

import logging
import time
import functools
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """Configuration for external guardrails."""
    
    # LLM Retry Guardrails
    llm_max_retries: int = 3
    llm_retry_delay: float = 1.0  # seconds
    llm_timeout: float = 30.0     # seconds per call
    
    # Loop Prevention Guardrails  
    max_replanning_events: int = 5
    max_same_task_attempts: int = 3
    max_total_execution_steps: int = 50
    
    # Memory Guardrails
    max_execution_steps_memory: int = 100
    max_context_summary_length: int = 4000  # chars
    
    @classmethod
    def from_env(cls) -> 'GuardrailConfig':
        """Create config from environment variables."""
        import os
        
        return cls(
            llm_max_retries=int(os.getenv('AEC_LLM_MAX_RETRIES', '3')),
            llm_retry_delay=float(os.getenv('AEC_LLM_RETRY_DELAY', '1.0')),
            llm_timeout=float(os.getenv('AEC_LLM_TIMEOUT', '30.0')),
            max_replanning_events=int(os.getenv('AEC_MAX_REPLANNING', '5')),
            max_same_task_attempts=int(os.getenv('AEC_MAX_TASK_ATTEMPTS', '3')),
            max_total_execution_steps=int(os.getenv('AEC_MAX_EXECUTION_STEPS', '50')),
            max_execution_steps_memory=int(os.getenv('AEC_MAX_MEMORY_STEPS', '100')),
            max_context_summary_length=int(os.getenv('AEC_MAX_CONTEXT_LENGTH', '4000'))
        )


class LLMRetryError(Exception):
    """Exception raised when LLM retry attempts are exhausted."""
    pass


class GuardrailViolationError(Exception):
    """Exception raised when execution guardrails are violated."""
    pass


def retry_llm_call(
    max_retries: int = 3,
    delay: float = 1.0,
    timeout: float = 30.0,
    exponential_backoff: bool = True
) -> Callable:
    """
    Simple decorator to retry LLM calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Base delay between retries in seconds
        timeout: Timeout for each individual call
        exponential_backoff: Use exponential backoff for delays
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Apply timeout to the function call
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    
                    # Check if call took too long (basic timeout check)
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.warning(f"LLM call took {elapsed:.2f}s, exceeding timeout of {timeout}s")
                    
                    # Success - log retry info if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(f"LLM call succeeded on attempt {attempt + 1}/{max_retries + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # If this was the last attempt, don't retry
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay for next attempt
                    if exponential_backoff:
                        retry_delay = delay * (2 ** attempt)  # 1s, 2s, 4s, etc.
                    else:
                        retry_delay = delay
                    
                    logger.warning(
                        f"LLM call failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    
                    time.sleep(retry_delay)
            
            # All retries exhausted
            logger.error(f"LLM call failed after {max_retries + 1} attempts")
            raise LLMRetryError(
                f"LLM call failed after {max_retries + 1} attempts. Last error: {last_exception}"
            ) from last_exception
            
        return wrapper
    return decorator


class ExecutionGuardrail:
    """Simple external execution guardrail with counter-based limits."""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset all counters."""
        self.replanning_count = 0
        self.total_steps = 0
        self.task_attempt_counts: Dict[str, int] = {}
        
    def check_replanning_limit(self) -> None:
        """Check if replanning limit has been exceeded."""
        if self.replanning_count > self.config.max_replanning_events:
            raise GuardrailViolationError(
                f"Maximum replanning events exceeded: {self.replanning_count}/{self.config.max_replanning_events}. "
                "The agent may be stuck in a replanning loop."
            )
            
    def check_execution_steps_limit(self) -> None:
        """Check if total execution steps limit has been exceeded."""
        if self.total_steps > self.config.max_total_execution_steps:
            raise GuardrailViolationError(
                f"Maximum execution steps exceeded: {self.total_steps}/{self.config.max_total_execution_steps}. "
                "The agent may be in an infinite execution loop."
            )
            
    def check_task_attempts_limit(self, task_id: str) -> None:
        """Check if individual task attempt limit has been exceeded."""
        attempts = self.task_attempt_counts.get(task_id, 0)
        if attempts > self.config.max_same_task_attempts:
            raise GuardrailViolationError(
                f"Maximum attempts for task '{task_id}' exceeded: {attempts}/{self.config.max_same_task_attempts}. "
                "The task may be failing repeatedly."
            )
    
    def record_replanning_event(self) -> None:
        """Record a replanning event."""
        self.replanning_count += 1
        logger.info(f"Replanning event recorded: {self.replanning_count}/{self.config.max_replanning_events}")
        self.check_replanning_limit()
        
    def record_execution_step(self) -> None:
        """Record an execution step."""
        self.total_steps += 1
        logger.debug(f"Execution step recorded: {self.total_steps}/{self.config.max_total_execution_steps}")
        self.check_execution_steps_limit()
        
    def record_task_attempt(self, task_id: str) -> None:
        """Record a task attempt."""
        current_attempts = self.task_attempt_counts.get(task_id, 0) + 1
        self.task_attempt_counts[task_id] = current_attempts
        logger.debug(f"Task attempt recorded for '{task_id}': {current_attempts}/{self.config.max_same_task_attempts}")
        self.check_task_attempts_limit(task_id)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current guardrail status."""
        return {
            "replanning_events": f"{self.replanning_count}/{self.config.max_replanning_events}",
            "total_steps": f"{self.total_steps}/{self.config.max_total_execution_steps}",
            "task_attempts": dict(self.task_attempt_counts),
            "limits_hit": {
                "replanning": self.replanning_count >= self.config.max_replanning_events,
                "execution_steps": self.total_steps >= self.config.max_total_execution_steps,
                "task_attempts": any(
                    count >= self.config.max_same_task_attempts 
                    for count in self.task_attempt_counts.values()
                )
            }
        }


class MemoryGuardrail:
    """Simple memory guardrail with FIFO cleanup."""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        
    def should_cleanup_memory(self, current_steps: int) -> bool:
        """Check if memory cleanup should be triggered."""
        return current_steps > self.config.max_execution_steps_memory
        
    def cleanup_execution_steps(self, steps: list) -> list:
        """Simple FIFO cleanup - keep only the most recent steps."""
        if len(steps) <= self.config.max_execution_steps_memory:
            return steps
            
        # Keep the most recent steps
        keep_count = self.config.max_execution_steps_memory
        cleaned_steps = steps[-keep_count:]
        
        removed_count = len(steps) - keep_count
        logger.info(f"Memory cleanup: removed {removed_count} old execution steps, keeping {keep_count} recent steps")
        
        return cleaned_steps
        
    def trim_context_summary(self, context_text: str) -> str:
        """Trim context summary to max length."""
        if len(context_text) <= self.config.max_context_summary_length:
            return context_text
            
        # Simple truncation with ellipsis
        trimmed = context_text[:self.config.max_context_summary_length - 3] + "..."
        
        logger.warning(
            f"Context summary trimmed from {len(context_text)} to {len(trimmed)} characters "
            f"(limit: {self.config.max_context_summary_length})"
        )
        
        return trimmed


class GuardrailMonitor:
    """Centralized monitoring and status reporting for all guardrails."""
    
    def __init__(self, execution_guardrail: ExecutionGuardrail, memory_guardrail: MemoryGuardrail):
        self.execution_guardrail = execution_guardrail
        self.memory_guardrail = memory_guardrail
        self.start_time = time.time()
        self.llm_retry_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retry_attempts": 0,
            "total_retry_time": 0.0
        }
    
    def record_llm_call(self, success: bool, retry_attempts: int = 0, retry_time: float = 0.0):
        """Record LLM call statistics."""
        self.llm_retry_stats["total_calls"] += 1
        if success:
            self.llm_retry_stats["successful_calls"] += 1
        else:
            self.llm_retry_stats["failed_calls"] += 1
        self.llm_retry_stats["retry_attempts"] += retry_attempts
        self.llm_retry_stats["total_retry_time"] += retry_time
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all guardrails."""
        uptime = time.time() - self.start_time
        
        execution_status = self.execution_guardrail.get_status()
        
        return {
            "monitoring": {
                "session_uptime": f"{uptime:.2f}s",
                "start_time": time.ctime(self.start_time)
            },
            "llm_performance": {
                "total_calls": self.llm_retry_stats["total_calls"],
                "success_rate": (
                    self.llm_retry_stats["successful_calls"] / self.llm_retry_stats["total_calls"]
                    if self.llm_retry_stats["total_calls"] > 0 else 0.0
                ),
                "retry_attempts": self.llm_retry_stats["retry_attempts"],
                "average_retry_time": (
                    self.llm_retry_stats["total_retry_time"] / self.llm_retry_stats["retry_attempts"]
                    if self.llm_retry_stats["retry_attempts"] > 0 else 0.0
                )
            },
            "execution_guardrails": execution_status,
            "memory_status": {
                "cleanup_triggered": "N/A - tracked per session",
                "context_trimming": "Active",
                "max_memory_steps": self.memory_guardrail.config.max_execution_steps_memory,
                "max_context_length": self.memory_guardrail.config.max_context_summary_length
            },
            "configuration": {
                "max_replanning_events": self.execution_guardrail.config.max_replanning_events,
                "max_task_attempts": self.execution_guardrail.config.max_same_task_attempts,
                "max_execution_steps": self.execution_guardrail.config.max_total_execution_steps,
                "llm_max_retries": self.execution_guardrail.config.llm_max_retries,
                "llm_retry_delay": f"{self.execution_guardrail.config.llm_retry_delay}s",
                "llm_timeout": f"{self.execution_guardrail.config.llm_timeout}s"
            }
        }
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Check for any guardrail alerts or warnings."""
        alerts = []
        
        execution_status = self.execution_guardrail.get_status()
        limits_hit = execution_status["limits_hit"]
        
        # Check for limits approaching
        replanning_ratio = (
            self.execution_guardrail.replanning_count / 
            self.execution_guardrail.config.max_replanning_events
        )
        if replanning_ratio >= 0.8:
            alerts.append({
                "level": "warning",
                "type": "replanning_limit_approaching", 
                "message": f"Replanning events at {replanning_ratio:.1%} of limit"
            })
        
        steps_ratio = (
            self.execution_guardrail.total_steps /
            self.execution_guardrail.config.max_total_execution_steps
        )
        if steps_ratio >= 0.8:
            alerts.append({
                "level": "warning",
                "type": "execution_steps_approaching",
                "message": f"Execution steps at {steps_ratio:.1%} of limit"
            })
        
        # Check LLM performance
        if self.llm_retry_stats["total_calls"] > 0:
            failure_rate = (
                self.llm_retry_stats["failed_calls"] / 
                self.llm_retry_stats["total_calls"]
            )
            if failure_rate > 0.2:  # More than 20% failure rate
                alerts.append({
                    "level": "warning",
                    "type": "high_llm_failure_rate",
                    "message": f"LLM failure rate: {failure_rate:.1%}"
                })
        
        return alerts


# Enhanced retry decorator with monitoring
def monitored_llm_retry(
    monitor: Optional[GuardrailMonitor] = None,
    max_retries: int = 3,
    delay: float = 1.0,
    timeout: float = 30.0
) -> Callable:
    """LLM retry decorator with monitoring integration."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            retry_attempts = 0
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful call
                    if monitor:
                        total_time = time.time() - start_time
                        monitor.record_llm_call(True, retry_attempts, total_time)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    retry_attempts += 1
                    
                    if attempt == max_retries:
                        break
                    
                    retry_delay = delay * (2 ** attempt)
                    time.sleep(retry_delay)
            
            # Record failed call
            if monitor:
                total_time = time.time() - start_time
                monitor.record_llm_call(False, retry_attempts, total_time)
            
            raise LLMRetryError(
                f"LLM call failed after {max_retries + 1} attempts. Last error: {last_exception}"
            ) from last_exception
            
        return wrapper
    return decorator


# Global default configuration (can be overridden)
default_guardrail_config = GuardrailConfig.from_env()

# Convenience function for creating retry decorator with default config
def default_llm_retry(func: Callable) -> Callable:
    """Apply default LLM retry configuration to a function."""
    config = default_guardrail_config
    return retry_llm_call(
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
        timeout=config.llm_timeout
    )(func)


def create_guardrail_dashboard() -> str:
    """Create a simple text-based dashboard of guardrail status."""
    config = GuardrailConfig.from_env()
    
    dashboard = f"""
╔══════════════════════════════════════════════════════════════╗
║                    GUARDRAIL CONFIGURATION                   ║
╠══════════════════════════════════════════════════════════════╣
║ LLM Retry Settings:                                          ║
║   Max Retries: {config.llm_max_retries:<3}   Delay: {config.llm_retry_delay:<4}s   Timeout: {config.llm_timeout:<4}s     ║
║                                                              ║
║ Execution Limits:                                            ║
║   Max Replanning Events: {config.max_replanning_events:<3}                               ║
║   Max Task Attempts: {config.max_same_task_attempts:<3}                                   ║
║   Max Total Steps: {config.max_total_execution_steps:<3}                                 ║
║                                                              ║
║ Memory Limits:                                               ║
║   Max Memory Steps: {config.max_execution_steps_memory:<3}                               ║
║   Max Context Length: {config.max_context_summary_length:<4} chars                     ║
║                                                              ║
║ Environment Variables:                                       ║
║   AEC_LLM_MAX_RETRIES, AEC_LLM_RETRY_DELAY                  ║
║   AEC_LLM_TIMEOUT, AEC_MAX_REPLANNING                       ║  
║   AEC_MAX_TASK_ATTEMPTS, AEC_MAX_EXECUTION_STEPS           ║
║   AEC_MAX_MEMORY_STEPS, AEC_MAX_CONTEXT_LENGTH             ║
╚══════════════════════════════════════════════════════════════╝
"""
    return dashboard.strip()