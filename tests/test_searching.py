import pytest
import os
import pandas as pd
from unittest.mock import Mock


class TestSearching:
    """Tests for the Searching functionality."""

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    def test_real_searcher_initialization(self, mock_searcher):
        """Test that the real searcher initializes properly."""
        assert mock_searcher is not None
        assert hasattr(mock_searcher, 'knowledge_search')
        assert hasattr(mock_searcher, 'db_version')
        assert hasattr(mock_searcher, 'last_updated')

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    @pytest.mark.parametrize("type", ["fts", "vector"])
    def test_basic_knowledge_search(self, mock_searcher, type):
        """Test basic knowledge search functionality."""
        results, info, plots = mock_searcher.knowledge_search(
            query="fatigue",
            type=type,
            modes=["0", "1", "2"],
            agencies=["TAIC"],
            document_type=["safety_issue"],
            year_range=(2010, 2024),
            limit=10
        )

        assert isinstance(results, pd.DataFrame)
        assert isinstance(info, dict)
        assert "total_results" in info
        assert len(results) <= 10

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    @pytest.mark.parametrize("type", ["fts", "vector"])
    def test_empty_serach(self, mock_searcher, type):
        """Test search that has no search query and simply a filter search"""
        results, info, plots = mock_searcher.knowledge_search(
            query="",
            type=type,
            modes=["0", "1", "2"],
            agencies=["TAIC"],
            document_type=["safety_issue"],
            year_range=(2007, 2010),
            limit=5
        )

        assert isinstance(results, pd.DataFrame)
        assert isinstance(info, dict)

        assert len(results) == 5


    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    @pytest.mark.parametrize("doc_type", ["safety_issue", "recommendation", "report_section"])
    def test_search_with_year_filtering(self, mock_searcher, doc_type):
        """Test search with year range filtering."""
        # Test different year ranges
        year_ranges = [(2007, 2015), (2015, 2024)]

        for year_range in year_ranges:
            results, info, plots = mock_searcher.knowledge_search(
                query="",
                document_type=[doc_type],
                type="fts",
                modes=["0", "1", "2"],
                agencies=["TAIC", "ATSB", "TSB"],
                year_range=year_range,
                limit=5
            )

            assert isinstance(results, pd.DataFrame)
            assert isinstance(info, dict)

            # Make sure that all results fall within the year range
            if not results.empty and 'year' in results.columns:
                assert results['year'].between(year_range[0], year_range[1]).all()

        

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    @pytest.mark.parametrize("relevance", [0.1, 0.5, 0.8])
    def test_search_relevance_filtering(self, mock_searcher, relevance):
        """Test search with different relevance thresholds."""
        results, info, plots = mock_searcher.knowledge_search(
            query="safety factor",
            type="vector",
            modes=["0", "1", "2"],
            agencies=["TAIC"],
            document_type=["safety_issue"],
            year_range=(2010, 2024),
            relevance=relevance,
            limit=10
        )

        assert isinstance(results, pd.DataFrame)
        assert isinstance(info, dict)

        assert "relevant_results" in info
        assert info["relevant_results"] <= info["total_results"]


        # Make sure that relevance is meaning results are no less than the limit
        assert (results['relevance'] >= relevance).all()

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    def test_empty_search_results(self, mock_searcher):
        """Test handling of searches that return no results."""
        results, info, plots = mock_searcher.knowledge_search(
            query="nonexistentterm12345",
            document_type=["safety_issue"],
            year_range=(2007, 2010),
            modes=["0", "1", "2"],
            agencies=["TAIC"],
            type="fts",
            limit=5
        )

        assert isinstance(results, pd.DataFrame)
        assert isinstance(info, dict)
        assert info["total_results"] == 0

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    def test_search_performance(self, mock_searcher):
        """Test that searches complete within reasonable time."""
        import time

        start_time = time.time()
        results, info, plots = mock_searcher.knowledge_search(
            query="safety factor",
            document_type=["safety_issue"],
            year_range=(2010, 2024),
            modes=["0", "1", "2"],
            agencies=["TAIC"],
            type="vector",
            limit=20
        )
        end_time = time.time()

        # Should complete within 30 seconds
        assert end_time - start_time < 45

    @pytest.mark.skipif(not os.getenv("TEST_USE_REAL_SERVICES"), reason="Requires real services")
    def test_large_result_set(self, mock_searcher):
        """Test handling of larger result sets."""
        results, info, plots = mock_searcher.knowledge_search(
            query="fatigue",
            document_type=["safety_issue", "recommendation"],
            year_range=(2007, 2024),
            modes=["0", "1", "2"],
            agencies=["TAIC", "ATSB", "TSB"],
            type="vector",
            limit=1000
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 1000
        assert isinstance(info, dict)
