# tests/test_core.py
"""
Day 1 tests — verify AdaptRouter wraps correctly.
These run WITHOUT making any API calls (the base router is mocked).
"""
import pytest
from unittest.mock import patch, MagicMock


class TestAdaptRouterInit:

    def test_import_succeeds(self):
        """AdaptRouter must be importable."""
        from adaptrouter import AdaptRouter
        assert AdaptRouter is not None

    def test_version_exists(self):
        """Package version must be defined."""
        import adaptrouter
        assert hasattr(adaptrouter, "__version__")
        assert adaptrouter.__version__ == "0.1.0"

    def test_domain_stored(self):
        """Domain tag must be stored on init."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter(domain="medical")
            assert router.domain == "medical"

    def test_session_id_generated(self):
        """Each AdaptRouter instance gets a unique session_id."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            r1 = AdaptRouter()
            r2 = AdaptRouter()
            assert r1.session_id != r2.session_id

    def test_session_id_is_short(self):
        """Session ID must be short enough to log — 8 characters."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            assert len(router.session_id) == 8


class TestRouteMethod:

    def test_route_adds_query_id(self):
        """Every result must have a query_id field."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            result = router.route("What is the capital of France?")
            assert "query_id" in result

    def test_query_id_is_unique_per_call(self):
        """Two route() calls must produce different query_ids."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router  = AdaptRouter()
            result1 = router.route("Query one")
            result2 = router.route("Query two")
            assert result1["query_id"] != result2["query_id"]

    def test_route_adds_domain_field(self):
        """Result must include domain tag."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter(domain="coding")
            result = router.route("test query")
            assert result["domain"] == "coding"

    def test_recent_decisions_capped_at_20(self):
        """In-memory decision history must never exceed 20 entries."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            for i in range(25):
                router.route(f"query number {i}")
            assert len(router._recent_decisions) <= 20

    def test_feedback_before_store_returns_memory_status(self):
        """feedback() before Day 2 store is set up returns graceful response."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            result   = router.route("test")
            feedback = router.feedback(result["query_id"], was_helpful=True)
            assert "status" in feedback
            assert feedback["query_id"] == result["query_id"]


class TestGetStats:

    def test_stats_returns_dict(self):
        """get_stats() must return a dictionary."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            stats  = router.get_stats()
            assert isinstance(stats, dict)

    def test_stats_has_required_keys(self):
        """Stats must include session_id, domain, mode."""
        from adaptrouter import AdaptRouter
        with patch("adaptrouter.core._BASE_AVAILABLE", False):
            router = AdaptRouter()
            stats  = router.get_stats()
            for key in ["session_id", "domain", "mode"]:
                assert key in stats, f"Missing key: {key}"