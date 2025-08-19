"""
Intelligent content management for research synthesis with token-aware processing.
"""

import tiktoken
from typing import List, Dict, Tuple, Optional
from loguru import logger

from akd.structures import SearchResultItem


class IntelligentContentManager:
    """
    Manages content size intelligently with token awareness and quality-based allocation.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_synthesis_tokens: int = 100000,  # Reserve 28k tokens for response + overhead
        debug: bool = False
    ):
        self.max_synthesis_tokens = max_synthesis_tokens
        self.debug = debug
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def estimate_total_tokens(self, results: List[SearchResultItem], context: str = "") -> int:
        """Estimate total tokens for synthesis input."""
        context_tokens = self.count_tokens(context)
        
        results_tokens = 0
        for result in results:
            # Count tokens for each field that goes into synthesis
            title_tokens = self.count_tokens(result.title or "")
            content_tokens = self.count_tokens(result.content or "")
            url_tokens = self.count_tokens(str(result.url))
            snippet_tokens = self.count_tokens(result.snippet or "")
            
            results_tokens += title_tokens + content_tokens + url_tokens + snippet_tokens + 50  # overhead
        
        return context_tokens + results_tokens
    
    def calculate_content_allocation(
        self, 
        results: List[SearchResultItem],
        target_tokens: int,
        quality_scores: Optional[List[float]] = None
    ) -> Dict[int, int]:
        """
        Calculate intelligent token allocation for each result based on quality and relevance.
        Returns a mapping of result_index -> allocated_tokens.
        """
        if not results:
            return {}
        
        # Calculate base allocation
        base_allocation = target_tokens // len(results)
        min_allocation = 200  # Minimum tokens per source
        max_allocation = 2000  # Maximum tokens per source
        
        # If no quality scores, use equal allocation
        if not quality_scores or len(quality_scores) != len(results):
            allocation = {}
            for i in range(len(results)):
                allocation[i] = max(min_allocation, min(base_allocation, max_allocation))
            return allocation
        
        # Quality-weighted allocation
        total_quality = sum(quality_scores)
        if total_quality == 0:
            return self.calculate_content_allocation(results, target_tokens, None)
        
        allocation = {}
        remaining_tokens = target_tokens
        
        for i, quality in enumerate(quality_scores):
            # Higher quality gets more tokens
            quality_weight = quality / total_quality
            allocated = int(target_tokens * quality_weight)
            
            # Apply bounds
            allocated = max(min_allocation, min(allocated, max_allocation))
            allocation[i] = allocated
            remaining_tokens -= allocated
        
        # Distribute remaining tokens to highest quality sources
        if remaining_tokens > 0:
            sorted_indices = sorted(range(len(quality_scores)), 
                                  key=lambda i: quality_scores[i], reverse=True)
            
            tokens_per_source = remaining_tokens // len(results)
            for i in sorted_indices:
                allocation[i] += tokens_per_source
                remaining_tokens -= tokens_per_source
                if remaining_tokens <= 0:
                    break
        
        return allocation
    
    def intelligent_content_reduction(
        self,
        results: List[SearchResultItem],
        target_tokens: int,
        quality_scores: Optional[List[float]] = None,
        context: str = ""
    ) -> List[SearchResultItem]:
        """
        Intelligently reduce content to fit within target token budget.
        """
        if not results:
            return results
        
        current_tokens = self.estimate_total_tokens(results, context)
        
        if self.debug:
            logger.debug(
                f"Content reduction: {current_tokens} tokens -> target {target_tokens} tokens"
            )
        
        # If already under target, return as-is
        if current_tokens <= target_tokens:
            if self.debug:
                logger.debug("Content already within target, no reduction needed")
            return results
        
        # Calculate token allocation per result
        allocation = self.calculate_content_allocation(results, target_tokens, quality_scores)
        
        reduced_results = []
        for i, result in enumerate(results):
            allocated_tokens = allocation.get(i, 500)  # fallback
            
            # Preserve essential metadata (small token cost)
            title_tokens = min(self.count_tokens(result.title or ""), 100)
            url_tokens = min(self.count_tokens(str(result.url)), 50)
            snippet_tokens = min(self.count_tokens(result.snippet or ""), 100)
            
            # Calculate remaining tokens for content
            remaining_for_content = allocated_tokens - title_tokens - url_tokens - snippet_tokens - 20  # overhead
            remaining_for_content = max(0, remaining_for_content)
            
            # Truncate content to fit allocation
            reduced_result = result.model_copy()
            if remaining_for_content > 0 and result.content:
                content_chars = len(result.content)
                if content_chars > 0:
                    # Estimate chars per token (rough approximation)
                    chars_per_token = content_chars / max(1, self.count_tokens(result.content))
                    target_chars = int(remaining_for_content * chars_per_token)
                    
                    if target_chars < content_chars:
                        # Intelligent truncation: try to keep complete sentences
                        truncated_content = self._smart_truncate(result.content, target_chars)
                        reduced_result.content = truncated_content
                        
                        if self.debug:
                            original_tokens = self.count_tokens(result.content)
                            new_tokens = self.count_tokens(truncated_content)
                            logger.debug(
                                f"Reduced content for {result.url}: {original_tokens} -> {new_tokens} tokens"
                            )
            else:
                reduced_result.content = ""
            
            reduced_results.append(reduced_result)
        
        final_tokens = self.estimate_total_tokens(reduced_results, context)
        if self.debug:
            reduction_ratio = final_tokens / current_tokens if current_tokens > 0 else 1.0
            logger.debug(
                f"Content reduction complete: {current_tokens} -> {final_tokens} tokens "
                f"(ratio: {reduction_ratio:.3f})"
            )
        
        return reduced_results
    
    def _smart_truncate(self, text: str, target_chars: int) -> str:
        """
        Intelligently truncate text to preserve sentence boundaries when possible.
        """
        if len(text) <= target_chars:
            return text
        
        # Try to find a good break point
        truncated = text[:target_chars]
        
        # Look for sentence ending within last 200 chars
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        best_break = -1
        
        search_start = max(0, target_chars - 200)
        search_text = text[search_start:target_chars]
        
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos > best_break:
                best_break = search_start + pos + len(ending) - 1
        
        if best_break > search_start:
            return text[:best_break]
        
        # Fallback: look for paragraph break
        para_break = truncated.rfind('\n\n')
        if para_break > target_chars * 0.7:  # Don't lose too much content
            return text[:para_break]
        
        # Final fallback: word boundary
        space_break = truncated.rfind(' ')
        if space_break > target_chars * 0.8:
            return text[:space_break] + "..."
        
        return truncated + "..."
    
    def manage_iteration_content(
        self,
        all_results: List[SearchResultItem],
        max_results_per_iteration: int = 20,
        max_total_results: int = 50
    ) -> List[SearchResultItem]:
        """
        Manage content accumulation across iterations to prevent memory leaks.
        """
        if len(all_results) <= max_total_results:
            return all_results
        
        if self.debug:
            logger.debug(
                f"Managing iteration content: {len(all_results)} results -> max {max_total_results}"
            )
        
        # Sort by relevance/quality if available, otherwise by recency
        # Keep most recent and highest quality results
        sorted_results = sorted(all_results, key=lambda x: (
            getattr(x, 'relevancy_score', 0.5),
            getattr(x, 'publication_date', ''),
        ), reverse=True)
        
        managed_results = sorted_results[:max_total_results]
        
        if self.debug:
            logger.debug(f"Kept {len(managed_results)} highest quality results")
        
        return managed_results