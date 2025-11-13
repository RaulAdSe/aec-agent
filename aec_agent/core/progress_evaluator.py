"""
Progress Evaluator - LLM-powered goal completion assessment for reactive reasoning.

This module evaluates whether the original goal has been achieved based on
accumulated execution context, task completion, and discovered information.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .reasoning_utils import ReasoningUtils, Task, TaskStatus

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import json


@dataclass
class ProgressAssessment:
    """Assessment of progress toward goal completion."""
    goal_achieved: bool
    confidence: float
    reasoning: str
    completion_percentage: float
    missing_requirements: List[str]
    evidence_for_completion: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ProgressEvaluator:
    """
    LLM-powered goal completion assessment for reactive reasoning.
    
    Evaluates whether the original goal has been achieved by analyzing
    execution context, task completion status, and accumulated evidence.
    """
    
    def __init__(self, llm=None):
        """Initialize the progress evaluator."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for intelligent progress evaluation
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=1500
            )
        else:
            self.llm = llm
        
        # Progress evaluation patterns for AEC domain
        self.evaluation_patterns = {
            "counting_tasks": {
                "completion_indicators": ["count", "total", "number", "quantity"],
                "evidence_requirements": ["numeric_result", "element_list"],
                "minimum_confidence": 0.8
            },
            "compliance_analysis": {
                "completion_indicators": ["compliant", "violation", "passes", "fails"],
                "evidence_requirements": ["validation_result", "rule_check"],
                "minimum_confidence": 0.9
            },
            "data_extraction": {
                "completion_indicators": ["extracted", "retrieved", "found", "loaded"],
                "evidence_requirements": ["data_output", "element_properties"],
                "minimum_confidence": 0.7
            },
            "calculation_tasks": {
                "completion_indicators": ["calculated", "computed", "measured"],
                "evidence_requirements": ["numeric_result", "calculation_output"],
                "minimum_confidence": 0.8
            },
            "search_tasks": {
                "completion_indicators": ["found", "located", "identified", "searched"],
                "evidence_requirements": ["search_results", "match_found"],
                "minimum_confidence": 0.7
            }
        }
    
    @traceable(name="evaluate_goal_progress", metadata={"component": "progress_evaluator"})
    def evaluate_progress(
        self,
        original_goal: str,
        execution_context: Dict[str, Any],
        current_tasks: List[Task]
    ) -> ProgressAssessment:
        """
        Evaluate progress toward achieving the original goal.
        
        Args:
            original_goal: The original goal statement
            execution_context: Rich context from execution memory
            current_tasks: Current list of tasks and their statuses
            
        Returns:
            ProgressAssessment with goal completion analysis
        """
        self.logger.info(f"Evaluating progress for goal: {original_goal[:50]}...")
        
        try:
            # First try LLM-powered intelligent evaluation
            llm_assessment = self._llm_evaluate_progress(original_goal, execution_context, current_tasks)
            if llm_assessment:
                self.logger.info(f"LLM assessment: {llm_assessment.confidence:.1%} confidence")
                return llm_assessment
            
            # Fallback to pattern-based evaluation
            pattern_assessment = self._pattern_based_evaluation(original_goal, execution_context, current_tasks)
            if pattern_assessment:
                self.logger.info(f"Pattern assessment: {pattern_assessment.confidence:.1%} confidence")
                return pattern_assessment
            
            # If all approaches fail, return conservative assessment
            return self._conservative_assessment(original_goal, execution_context, current_tasks)
            
        except Exception as e:
            self.logger.error(f"Progress evaluation failed: {e}")
            return self._conservative_assessment(original_goal, execution_context, current_tasks)
    
    @traceable(name="llm_progress_evaluation")
    def _llm_evaluate_progress(
        self,
        original_goal: str,
        execution_context: Dict[str, Any],
        current_tasks: List[Task]
    ) -> Optional[ProgressAssessment]:
        """Use LLM to intelligently evaluate goal progress."""
        
        # Prepare context summary for LLM
        context_summary = self._prepare_progress_context(execution_context)
        task_summary = self._prepare_task_summary(current_tasks)
        
        # Create evaluation prompt
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AEC (Architecture, Engineering, Construction) compliance analyst.

Your job is to evaluate whether a goal has been achieved based on execution context and evidence.

Task Analysis Guidelines:
1. A goal is achieved when the specific request has been fulfilled with sufficient evidence
2. For counting tasks: Need actual numeric results (e.g., "5 doors found")
3. For analysis tasks: Need clear conclusions or findings
4. For data tasks: Need evidence that data was successfully loaded/processed
5. For compliance tasks: Need validation results or compliance status
6. For search tasks: Need evidence of what was found or not found

Original Goal: {goal}
Execution Context: {execution_context}
Task Status Summary: {task_summary}

Based on this information, evaluate whether the original goal has been achieved.

Return ONLY a JSON response with:
{{
  "goal_achieved": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Clear explanation of why goal is/isn't achieved",
  "completion_percentage": 0-100,
  "missing_requirements": ["list of what's still needed if goal not achieved"],
  "evidence_for_completion": ["list of evidence supporting completion"],
  "recommendations": ["list of next steps if needed"]
}}

Be strict but fair - only mark as achieved if there's clear evidence the goal was met."""),
            ("human", "Please evaluate the goal progress:")
        ])
        
        try:
            # Execute LLM evaluation
            chain = evaluation_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "goal": original_goal,
                "execution_context": context_summary,
                "task_summary": task_summary
            })
            
            # Parse JSON response
            eval_data = json.loads(response.strip())
            
            if not isinstance(eval_data, dict):
                self.logger.warning("LLM returned invalid evaluation format")
                return None
            
            return ProgressAssessment(
                goal_achieved=eval_data.get("goal_achieved", False),
                confidence=eval_data.get("confidence", 0.0),
                reasoning=eval_data.get("reasoning", "LLM evaluation completed"),
                completion_percentage=eval_data.get("completion_percentage", 0),
                missing_requirements=eval_data.get("missing_requirements", []),
                evidence_for_completion=eval_data.get("evidence_for_completion", []),
                recommendations=eval_data.get("recommendations", []),
                metadata={"method": "llm_reasoning", "llm_response": eval_data}
            )
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM evaluation response was not valid JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            return None
    
    def _pattern_based_evaluation(
        self,
        original_goal: str,
        execution_context: Dict[str, Any],
        current_tasks: List[Task]
    ) -> Optional[ProgressAssessment]:
        """Fallback pattern-based evaluation when LLM fails."""
        
        goal_lower = original_goal.lower()
        
        # Detect goal type
        goal_type = None
        for pattern_type, pattern_config in self.evaluation_patterns.items():
            indicators = pattern_config["completion_indicators"]
            if any(indicator in goal_lower for indicator in indicators):
                goal_type = pattern_type
                break
        
        if not goal_type:
            return None
        
        # Apply pattern-based evaluation
        pattern_config = self.evaluation_patterns[goal_type]
        
        # Check task completion
        completed_tasks = [task for task in current_tasks if task.status == TaskStatus.COMPLETED]
        total_tasks = len(current_tasks)
        completion_percentage = (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0
        
        # Check for required evidence
        evidence_requirements = pattern_config["evidence_requirements"]
        discovered_context = execution_context.get("discovered_context_summary", {})
        
        evidence_found = []
        missing_evidence = []
        
        for requirement in evidence_requirements:
            if requirement in str(discovered_context).lower():
                evidence_found.append(f"Found {requirement} in execution context")
            else:
                missing_evidence.append(requirement)
        
        # Determine goal achievement
        minimum_confidence = pattern_config["minimum_confidence"]
        has_evidence = len(evidence_found) > 0
        high_completion = completion_percentage >= 80
        
        goal_achieved = has_evidence and high_completion
        confidence = min(minimum_confidence, completion_percentage / 100.0) if goal_achieved else 0.3
        
        reasoning = f"Pattern-based evaluation for {goal_type}: "
        reasoning += f"{completion_percentage:.0f}% tasks completed, "
        reasoning += f"evidence found: {len(evidence_found)}/{len(evidence_requirements)}"
        
        return ProgressAssessment(
            goal_achieved=goal_achieved,
            confidence=confidence,
            reasoning=reasoning,
            completion_percentage=completion_percentage,
            missing_requirements=missing_evidence,
            evidence_for_completion=evidence_found,
            recommendations=["Continue with remaining tasks"] if not goal_achieved else ["Goal achieved"],
            metadata={"method": "pattern_based", "goal_type": goal_type}
        )
    
    def _conservative_assessment(
        self,
        original_goal: str,
        execution_context: Dict[str, Any],
        current_tasks: List[Task]
    ) -> ProgressAssessment:
        """Conservative fallback assessment when all other methods fail."""
        
        completed_tasks = [task for task in current_tasks if task.status == TaskStatus.COMPLETED]
        total_tasks = len(current_tasks)
        
        completion_percentage = (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0
        
        # Conservative: only mark as achieved if ALL tasks are completed
        goal_achieved = completion_percentage >= 100
        confidence = 0.5 if goal_achieved else 0.2
        
        reasoning = f"Conservative evaluation: {len(completed_tasks)}/{total_tasks} tasks completed"
        
        missing_requirements = []
        if not goal_achieved:
            pending_tasks = [task for task in current_tasks if task.status != TaskStatus.COMPLETED]
            missing_requirements = [f"Complete task: {task.name}" for task in pending_tasks[:3]]
        
        return ProgressAssessment(
            goal_achieved=goal_achieved,
            confidence=confidence,
            reasoning=reasoning,
            completion_percentage=completion_percentage,
            missing_requirements=missing_requirements,
            evidence_for_completion=[f"Completed {len(completed_tasks)} tasks"],
            recommendations=["Complete remaining tasks"] if not goal_achieved else ["All tasks completed"],
            metadata={"method": "conservative_fallback"}
        )
    
    def _prepare_progress_context(self, execution_context: Dict[str, Any]) -> str:
        """Prepare execution context summary for progress evaluation."""
        
        summary_parts = []
        
        # Execution statistics
        exec_summary = execution_context.get("execution_summary", {})
        if exec_summary:
            total_steps = exec_summary.get("total_execution_steps", 0)
            success_rate = exec_summary.get("success_rate", 0.0)
            summary_parts.append(f"Execution: {total_steps} steps with {success_rate:.1%} success rate")
        
        # Task completion info
        completed_tasks = exec_summary.get("unique_tasks_completed", 0)
        attempted_tasks = exec_summary.get("unique_tasks_attempted", 0)
        if attempted_tasks > 0:
            task_rate = completed_tasks / attempted_tasks
            summary_parts.append(f"Task completion: {completed_tasks}/{attempted_tasks} ({task_rate:.1%})")
        
        # Recent validation confidence
        recent_confidence = execution_context.get("recent_validation_confidence", [])
        if recent_confidence:
            avg_confidence = sum(recent_confidence) / len(recent_confidence)
            summary_parts.append(f"Recent validation confidence: {avg_confidence:.1%}")
        
        # Discovered context
        discovered_context = execution_context.get("discovered_context_summary", {})
        if discovered_context:
            summary_parts.append("Discovered context:")
            for key, value in list(discovered_context.items())[:5]:  # Show top 5
                summary_parts.append(f"  • {key}: {str(value)[:50]}")
        
        # Plan modifications
        plan_mods = execution_context.get("plan_modifications_count", 0)
        if plan_mods > 0:
            summary_parts.append(f"Plan modifications: {plan_mods} (indicates adaptive execution)")
        
        return "\n".join(summary_parts)
    
    def _prepare_task_summary(self, current_tasks: List[Task]) -> str:
        """Prepare task status summary for progress evaluation."""
        
        if not current_tasks:
            return "No tasks in current plan"
        
        status_counts = {}
        for task in current_tasks:
            status = task.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary_parts = [f"Total tasks: {len(current_tasks)}"]
        
        for status, count in status_counts.items():
            summary_parts.append(f"  • {status.value}: {count}")
        
        # Show some task details
        completed_tasks = [task for task in current_tasks if task.status == TaskStatus.COMPLETED]
        if completed_tasks:
            summary_parts.append("Recent completed tasks:")
            for task in completed_tasks[-3:]:  # Show last 3
                summary_parts.append(f"  ✓ {task.name}")
        
        failed_tasks = [task for task in current_tasks if task.status == TaskStatus.FAILED]
        if failed_tasks:
            summary_parts.append("Failed tasks:")
            for task in failed_tasks[-2:]:  # Show last 2
                summary_parts.append(f"  ✗ {task.name}")
        
        return "\n".join(summary_parts)
    
    def should_continue_execution(self, assessment: ProgressAssessment, max_iterations: int, current_iteration: int) -> bool:
        """
        Determine if execution should continue based on progress assessment.
        
        Args:
            assessment: Current progress assessment
            max_iterations: Maximum allowed iterations
            current_iteration: Current iteration number
            
        Returns:
            True if execution should continue
        """
        # Stop if goal is achieved with high confidence
        if assessment.goal_achieved and assessment.confidence >= 0.7:
            self.logger.info("Stopping: Goal achieved with high confidence")
            return False
        
        # Stop if reached max iterations
        if current_iteration >= max_iterations:
            self.logger.info("Stopping: Maximum iterations reached")
            return False
        
        # Stop if very low progress and low confidence
        if assessment.completion_percentage < 20 and assessment.confidence < 0.3 and current_iteration > 5:
            self.logger.info("Stopping: Low progress and confidence after multiple iterations")
            return False
        
        # Continue otherwise
        return True
    
    def get_evaluation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get available evaluation patterns for different goal types."""
        return self.evaluation_patterns.copy()
    
    def validate_assessment(self, assessment: ProgressAssessment) -> bool:
        """
        Validate that an assessment is logically consistent.
        
        Args:
            assessment: The assessment to validate
            
        Returns:
            True if the assessment is valid
        """
        # Check confidence is in valid range
        if not (0.0 <= assessment.confidence <= 1.0):
            self.logger.warning(f"Invalid confidence: {assessment.confidence}")
            return False
        
        # Check completion percentage is in valid range
        if not (0 <= assessment.completion_percentage <= 100):
            self.logger.warning(f"Invalid completion percentage: {assessment.completion_percentage}")
            return False
        
        # Check logical consistency
        if assessment.goal_achieved and assessment.completion_percentage < 50:
            self.logger.warning("Goal marked as achieved but completion percentage is low")
            return False
        
        if not assessment.goal_achieved and assessment.confidence > 0.9:
            self.logger.warning("Goal not achieved but confidence is very high")
            return False
        
        return True