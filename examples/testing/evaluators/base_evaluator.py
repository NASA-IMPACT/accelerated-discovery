"""
Base evaluator class for LLM-as-judge evaluations.

This module provides the foundation for all evaluation components,
implementing the core LLM-based comparison logic.
"""

import asyncio
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Import LLM client
import openai
from loguru import logger
from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).parent.parent))

from test_models import EvaluationResult


class BaseLLMEvaluator(ABC):
    """
    Base class for LLM-based evaluators.

    This class provides the common functionality for making LLM calls
    and parsing evaluation results in a consistent format.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            model: LLM model to use for evaluation
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            debug: Enable debug logging
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug = debug

        # Initialize OpenAI client
        self.client = AsyncOpenAI()

        logger.info(f"Initialized {self.__class__.__name__} with model {model}")

    async def evaluate(
        self,
        actual_output: Any,
        expected_output: Any,
        context: Dict[str, Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate actual output against expected output.

        Args:
            actual_output: The actual output from the component
            expected_output: The expected output to compare against
            context: Additional context for evaluation

        Returns:
            EvaluationResult with pass/fail decision and reasoning
        """
        logger.info(f"Starting evaluation with {self.__class__.__name__}")

        try:
            # Format the evaluation prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._format_user_prompt(
                actual_output,
                expected_output,
                context or {},
            )

            if self.debug:
                logger.debug(f"System prompt: {system_prompt[:200]}...")
                logger.debug(f"User prompt: {user_prompt[:200]}...")

            # Make LLM call with retries
            response = await self._make_llm_call_with_retries(
                system_prompt,
                user_prompt,
            )

            # Parse the response
            evaluation = self._parse_llm_response(response)

            # Add metadata
            evaluation.component = self._get_component_name()
            evaluation.evaluator_model = self.model
            evaluation.evaluation_time = datetime.now()

            logger.info(
                f"Evaluation completed: {'PASS' if evaluation.passed else 'FAIL'} ({evaluation.confidence:.2f})",
            )

            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

            # Return failure result
            return EvaluationResult(
                passed=False,
                confidence=0.0,
                reasoning=f"Evaluation failed due to error: {e}",
                specific_feedback={"error": str(e)},
                component=self._get_component_name(),
                evaluator_model=self.model,
                evaluation_time=datetime.now(),
            )

    async def _make_llm_call_with_retries(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make LLM call with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_tokens=1000,
                )

                return response.choices[0].message.content.strip()

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e

            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e

        raise last_exception

    def _parse_llm_response(self, response: str) -> EvaluationResult:
        """
        Parse LLM response into EvaluationResult.

        Expected format:
        DECISION: PASS/FAIL
        CONFIDENCE: 0.85
        REASONING: Detailed explanation...
        """
        lines = response.strip().split("\n")
        decision = None
        confidence = 0.0
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("DECISION:"):
                decision_text = line.replace("DECISION:", "").strip().upper()
                decision = decision_text in ["PASS", "TRUE", "YES"]

            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except ValueError:
                    confidence = 0.5  # Default confidence

            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        # If no explicit decision found, try to infer from the response
        if decision is None:
            response_lower = response.lower()
            if any(
                word in response_lower
                for word in ["pass", "yes", "true", "correct", "match"]
            ):
                decision = True
            else:
                decision = False

        # Use full response as reasoning if no explicit reasoning found
        if not reasoning:
            reasoning = response

        return EvaluationResult(
            passed=decision,
            confidence=confidence,
            reasoning=reasoning,
            specific_feedback={},
            component="",  # Will be set by caller
            evaluator_model=self.model,
        )

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this evaluator.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def _format_user_prompt(
        self,
        actual_output: Any,
        expected_output: Any,
        context: Dict[str, Any],
    ) -> str:
        """
        Format the user prompt for evaluation.

        Args:
            actual_output: Actual output from component
            expected_output: Expected output
            context: Additional context

        Returns:
            Formatted user prompt string
        """
        pass

    @abstractmethod
    def _get_component_name(self) -> str:
        """
        Get the name of the component this evaluator is for.

        Returns:
            Component name string
        """
        pass


class BinaryEvaluator(BaseLLMEvaluator):
    """
    Simple binary evaluator for basic pass/fail decisions.

    This evaluator provides a simple framework for binary evaluation
    tasks where components just need to be evaluated as pass or fail.
    """

    def __init__(
        self,
        component_name: str,
        evaluation_criteria: str,
        **kwargs,
    ):
        """
        Initialize binary evaluator.

        Args:
            component_name: Name of component being evaluated
            evaluation_criteria: Criteria for evaluation
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.component_name = component_name
        self.evaluation_criteria = evaluation_criteria

    def _get_system_prompt(self) -> str:
        """Get system prompt for binary evaluation."""
        return f"""You are an expert evaluator for the {self.component_name} component.

Your task is to evaluate whether the actual output adequately matches the expected output based on these criteria:

{self.evaluation_criteria}

You must respond in this exact format:
DECISION: PASS or FAIL
CONFIDENCE: A number between 0.0 and 1.0
REASONING: Detailed explanation of your decision

Be objective and focus on the specific criteria. Consider semantic similarity, not just exact matches."""

    def _format_user_prompt(
        self,
        actual_output: Any,
        expected_output: Any,
        context: Dict[str, Any],
    ) -> str:
        """Format user prompt for binary evaluation."""
        context_str = ""
        if context:
            context_items = [f"- {k}: {v}" for k, v in context.items()]
            context_str = "\n\nContext:\n" + "\n".join(context_items)

        return f"""Please evaluate the following output:

Expected Output:
{self._format_output_for_prompt(expected_output)}

Actual Output:
{self._format_output_for_prompt(actual_output)}{context_str}

Does the actual output adequately match the expected output according to the evaluation criteria?"""

    def _format_output_for_prompt(self, output: Any) -> str:
        """Format output for inclusion in prompt."""
        if isinstance(output, str):
            return output
        elif isinstance(output, (list, dict)):
            import json

            return json.dumps(output, indent=2, default=str)
        else:
            return str(output)

    def _get_component_name(self) -> str:
        """Get component name."""
        return self.component_name
