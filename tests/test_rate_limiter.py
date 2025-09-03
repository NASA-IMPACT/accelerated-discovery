#!/usr/bin/env python3
"""
Unit tests for the RateLimiter class in akd.utils.

These tests verify that the RateLimiter correctly enforces rate limits
for API requests in various scenarios.
"""

import asyncio
import time

import pytest

from akd.utils import RateLimiter


class TestRateLimiterBasicFunctionality:
    """Test basic RateLimiter functionality."""

    def test_rate_limiter_creation(self):
        """Test RateLimiter initializes correctly with different rates."""
        # Test default rate (1.0 req/sec)
        limiter = RateLimiter()
        assert limiter.max_calls_per_second == 1.0
        assert limiter.min_interval == 1.0
        assert limiter.last_called == 0.0

        # Test custom rate (2.0 req/sec)
        limiter2 = RateLimiter(max_calls_per_second=2.0)
        assert limiter2.max_calls_per_second == 2.0
        assert limiter2.min_interval == 0.5

        # Test slow rate (0.5 req/sec)
        limiter3 = RateLimiter(max_calls_per_second=0.5)
        assert limiter3.max_calls_per_second == 0.5
        assert limiter3.min_interval == 2.0

    @pytest.mark.asyncio
    async def test_single_request_immediate(self):
        """Test that the first request goes through immediately."""
        limiter = RateLimiter(max_calls_per_second=1.0)

        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time

        # First request should be immediate (< 0.1s tolerance)
        assert elapsed < 0.1
        assert limiter.last_called > 0  # Should be updated

    @pytest.mark.asyncio
    async def test_rate_limiting_delay(self):
        """Test that second quick request is properly delayed."""
        limiter = RateLimiter(max_calls_per_second=1.0)  # 1 second interval

        # First request - immediate
        start_time = time.time()
        await limiter.acquire()
        first_request_time = time.time() - start_time

        # Second request immediately after - should be delayed
        await limiter.acquire()
        total_time = time.time() - start_time

        assert first_request_time < 0.1  # First was immediate
        assert 0.9 <= total_time <= 1.2  # Second was delayed ~1 second (with tolerance)

    @pytest.mark.asyncio
    async def test_timing_accuracy(self):
        """Test that delays are approximately correct."""
        limiter = RateLimiter(max_calls_per_second=2.0)  # 0.5 second interval

        start_time = time.time()

        # Make 3 requests in quick succession
        await limiter.acquire()  # Immediate
        await limiter.acquire()  # Delayed ~0.5s
        await limiter.acquire()  # Delayed another ~0.5s

        total_time = time.time() - start_time

        # Should take ~1 second total (0 + 0.5 + 0.5) with tolerance
        assert 0.8 <= total_time <= 1.3


class TestRateLimiterConfiguration:
    """Test RateLimiter with different rate configurations."""

    @pytest.mark.asyncio
    async def test_different_rates(self):
        """Test various max_calls_per_second values."""
        test_cases = [
            (0.5, 2.0),  # 0.5 req/sec = 2 second intervals
            (1.0, 1.0),  # 1.0 req/sec = 1 second intervals
            (2.0, 0.5),  # 2.0 req/sec = 0.5 second intervals
        ]

        for rate, expected_interval in test_cases:
            limiter = RateLimiter(max_calls_per_second=rate)
            assert limiter.min_interval == expected_interval

            # Test actual timing for faster rates only (to keep tests quick)
            if rate >= 2.0:
                start_time = time.time()
                await limiter.acquire()  # Immediate
                await limiter.acquire()  # Delayed
                elapsed = time.time() - start_time

                # Should be close to expected interval (with tolerance)
                assert (expected_interval - 0.1) <= elapsed <= (expected_interval + 0.3)

    def test_min_interval_calculation(self):
        """Test that min_interval is calculated correctly."""
        test_cases = [
            (0.1, 10.0),
            (0.5, 2.0),
            (1.0, 1.0),
            (2.0, 0.5),
            (5.0, 0.2),
        ]

        for rate, expected_interval in test_cases:
            limiter = RateLimiter(max_calls_per_second=rate)
            assert abs(limiter.min_interval - expected_interval) < 0.001


class TestRateLimiterTiming:
    """Test RateLimiter timing behavior."""

    @pytest.mark.asyncio
    async def test_no_delay_after_long_gap(self):
        """Test that requests after long delays are immediate."""
        limiter = RateLimiter(max_calls_per_second=1.0)

        # First request
        await limiter.acquire()

        # Wait longer than the interval
        await asyncio.sleep(1.1)

        # Second request should be immediate since enough time passed
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time

        assert elapsed < 0.1  # Should be immediate

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self):
        """Test pattern of multiple requests in sequence."""
        limiter = RateLimiter(max_calls_per_second=4.0)  # 0.25 second intervals

        start_time = time.time()
        request_times = []

        # Make 4 requests
        for i in range(4):
            await limiter.acquire()
            request_times.append(time.time() - start_time)

        # Verify timing pattern
        assert request_times[0] < 0.1  # First immediate
        assert 0.2 <= request_times[1] <= 0.4  # ~0.25s
        assert 0.4 <= request_times[2] <= 0.7  # ~0.5s
        assert 0.6 <= request_times[3] <= 1.0  # ~0.75s


class TestRateLimiterConcurrency:
    """Test RateLimiter with concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test multiple async tasks sharing same limiter."""
        limiter = RateLimiter(max_calls_per_second=2.0)  # 0.5 second intervals
        request_times = []
        start_time = time.time()

        async def make_request(task_id):
            await limiter.acquire()
            request_times.append((task_id, time.time() - start_time))

        # Start 3 tasks concurrently
        tasks = [make_request(i) for i in range(3)]
        await asyncio.gather(*tasks)

        # Sort by completion time
        request_times.sort(key=lambda x: x[1])

        # Verify they were spaced out correctly
        assert request_times[0][1] < 0.1  # First immediate
        assert 0.4 <= request_times[1][1] <= 0.7  # Second delayed ~0.5s
        assert 0.8 <= request_times[2][1] <= 1.3  # Third delayed ~1.0s total

    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Test that the asyncio.Lock works correctly."""
        limiter = RateLimiter(max_calls_per_second=1.0)
        results = []

        async def worker(worker_id):
            await limiter.acquire()
            # Simulate some work
            await asyncio.sleep(0.01)
            results.append(worker_id)

        # Start multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All workers should complete (no deadlocks)
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}


class TestRateLimiterIntegration:
    """Test RateLimiter integration scenarios."""

    @pytest.mark.asyncio
    async def test_with_mock_api_calls(self):
        """Test RateLimiter with simulated API calls."""
        limiter = RateLimiter(max_calls_per_second=1.0)
        api_call_times = []

        # Mock API function
        async def mock_api_call(data):
            await limiter.acquire()  # Rate limiting
            api_call_times.append(time.time())
            # Simulate API response time
            await asyncio.sleep(0.01)
            return f"response_for_{data}"

        start_time = time.time()

        # Make multiple API calls
        tasks = [mock_api_call(f"data_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 3
        assert all("response_for_data_" in result for result in results)

        # Verify rate limiting worked (calls spaced ~1 second apart)
        total_time = time.time() - start_time
        assert 1.8 <= total_time <= 2.5  # ~2 seconds with tolerance

        # Verify call timing
        relative_times = [t - api_call_times[0] for t in api_call_times]
        assert relative_times[0] < 0.1  # First immediate
        assert 0.9 <= relative_times[1] <= 1.2  # Second ~1s later
        assert 1.8 <= relative_times[2] <= 2.3  # Third ~2s later

    @pytest.mark.asyncio
    async def test_semantic_scholar_integration(self):
        """Test RateLimiter works with SemanticScholarSearchTool pattern."""
        from akd.tools.search.semantic_scholar_search import (
            SemanticScholarSearchTool,
            SemanticScholarSearchToolConfig,
        )

        # Create tool with fast rate limit for testing
        config = SemanticScholarSearchToolConfig(
            requests_per_second=2.0,  # 0.5 second intervals
            debug=False,
        )
        tool = SemanticScholarSearchTool(config=config, debug=False)

        # Verify rate limiter was created correctly
        assert tool.rate_limiter is not None
        assert tool.rate_limiter.max_calls_per_second == 2.0
        assert tool.rate_limiter.min_interval == 0.5

        # Test the acquire method works
        start_time = time.time()
        await tool.rate_limiter.acquire()  # Immediate
        await tool.rate_limiter.acquire()  # Delayed ~0.5s
        elapsed = time.time() - start_time

        assert 0.4 <= elapsed <= 0.8  # ~0.5s with tolerance


# Utility test for edge cases
class TestRateLimiterEdgeCases:
    """Test RateLimiter edge cases and error conditions."""

    def test_zero_rate_protection(self):
        """Test that zero or negative rates are handled."""
        # This would cause division by zero in min_interval calculation
        # The RateLimiter should handle this gracefully or the config should prevent it
        try:
            limiter = RateLimiter(max_calls_per_second=0.0)
            # If it doesn't raise an error, min_interval should be inf
            assert limiter.min_interval == float("inf")
        except (ValueError, ZeroDivisionError):
            # It's also acceptable to raise an error for invalid rates
            pass

    @pytest.mark.asyncio
    async def test_very_high_rate(self):
        """Test with very high rate limit."""
        limiter = RateLimiter(max_calls_per_second=100.0)  # Very fast

        start_time = time.time()

        # Multiple quick requests should all be nearly immediate
        for _ in range(5):
            await limiter.acquire()

        elapsed = time.time() - start_time
        assert elapsed < 0.2  # Should be very fast

    @pytest.mark.asyncio
    async def test_very_slow_rate(self):
        """Test with very slow rate limit (for functionality, not full timing)."""
        limiter = RateLimiter(max_calls_per_second=0.1)  # 10 second intervals

        # Just test the first request is immediate and setup is correct
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time

        assert elapsed < 0.1  # First should be immediate
        assert limiter.min_interval == 10.0  # Correct interval calculated


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
