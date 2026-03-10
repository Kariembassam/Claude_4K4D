"""
Tests for core.base_node
==========================
Validates the BaseEasyVolcapNode abstract class: safe execution,
dataset_info validation, error formatting, and immutable updates.
"""

import copy
from unittest.mock import patch, MagicMock

import pytest

from core.base_node import BaseEasyVolcapNode
from core.constants import create_empty_dataset_info, DATASET_INFO_TYPE


# ---------------------------------------------------------------------------
# Concrete test subclass
# ---------------------------------------------------------------------------
class _TestNode(BaseEasyVolcapNode):
    """Minimal concrete node for testing the abstract base class."""

    CATEGORY = "4K4D/Test"
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE,)
    RETURN_NAMES = ("dataset_info",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            }
        }

    def _safe_execute(self, dataset_info, **kwargs):
        """Simple pass-through that adds a marker."""
        updated = self._update_dataset_info(dataset_info, test_marker=True)
        return (updated,)


class _CrashingNode(BaseEasyVolcapNode):
    """Node that raises an exception inside _safe_execute."""

    CATEGORY = "4K4D/Test"
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE,)
    RETURN_NAMES = ("dataset_info",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            }
        }

    def _safe_execute(self, dataset_info, **kwargs):
        raise RuntimeError("Something went terribly wrong!")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBaseEasyVolcapNode:
    """Validate BaseEasyVolcapNode contract."""

    def test_safe_execute_success(self, sample_dataset_info):
        """A normal node execution should return updated dataset_info."""
        node = _TestNode()
        result = node.execute(dataset_info=sample_dataset_info)

        assert isinstance(result, tuple)
        assert len(result) == 1
        updated_info = result[0]
        assert updated_info.get("test_marker") is True

    def test_safe_execute_catches_exceptions(self, sample_dataset_info):
        """A crashing node should NOT raise; it should return an error in dataset_info."""
        node = _CrashingNode()
        # The execute() method wraps _safe_execute in try/except
        result = node.execute(dataset_info=sample_dataset_info)

        assert isinstance(result, tuple)
        updated_info = result[0]
        # Should contain error information, not crash
        assert isinstance(updated_info, dict)
        # The base class should have caught the exception and logged it
        errors = updated_info.get("errors", [])
        assert len(errors) > 0 or "error" in str(updated_info).lower()

    def test_validate_dataset_info_valid(self, sample_dataset_info):
        """A well-formed dataset_info should pass validation."""
        node = _TestNode()
        # Should not raise
        node._validate_dataset_info(sample_dataset_info)

    def test_validate_dataset_info_missing_keys(self):
        """An empty dict should fail validation."""
        node = _TestNode()
        with pytest.raises((ValueError, KeyError, Exception)):
            node._validate_dataset_info({})

    def test_update_dataset_info_returns_new_dict(self, sample_dataset_info):
        """_update_dataset_info should return a NEW dict, not mutate the original."""
        node = _TestNode()
        original_copy = copy.deepcopy(sample_dataset_info)

        updated = node._update_dataset_info(sample_dataset_info, new_key="new_value")

        # Original should be unmodified
        assert "new_key" not in sample_dataset_info or sample_dataset_info == original_copy
        # Updated should have the new key
        assert updated.get("new_key") == "new_value"
        # Updated should be a different object
        assert updated is not sample_dataset_info

    def test_format_user_error(self):
        """_format_user_error should produce a readable string."""
        node = _TestNode()
        try:
            raise ValueError("bad input path")
        except ValueError as e:
            msg = node._format_user_error(e)

        assert isinstance(msg, str)
        assert "bad input path" in msg
        assert len(msg) > 0

    def test_input_types_has_required(self):
        """INPUT_TYPES should include 'required' key with dataset_info."""
        input_types = _TestNode.INPUT_TYPES()
        assert "required" in input_types
        assert "dataset_info" in input_types["required"]

    def test_node_has_category(self):
        """All nodes must declare a CATEGORY."""
        assert hasattr(_TestNode, "CATEGORY")
        assert _TestNode.CATEGORY == "4K4D/Test"

    def test_node_has_function(self):
        """All nodes must declare a FUNCTION name."""
        assert hasattr(_TestNode, "FUNCTION")
        assert _TestNode.FUNCTION == "execute"
