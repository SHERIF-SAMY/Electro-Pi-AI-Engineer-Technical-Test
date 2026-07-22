import json
import pytest

from agent.tools import get_order_status





@pytest.mark.asyncio
async def test_tool_returns_dict():
    data = await get_order_status.ainvoke({"order_id": "ORD-001"})
    assert isinstance(data, dict)
    assert data["order_id"] == "ORD-001"


@pytest.mark.asyncio
async def test_tool_handles_invalid_id_gracefully():
    data = await get_order_status.ainvoke({"order_id": "BAD-ID"})
    assert "error" in data
    assert data["code"] == 404


@pytest.mark.asyncio
async def test_tool_handles_empty_id():
    data = await get_order_status.ainvoke({"order_id": ""})
    assert data["code"] == 400


@pytest.mark.asyncio
async def test_tool_response_structure():
    data = await get_order_status.ainvoke({"order_id": "ORD-003"})
    expected_keys = {"order_id", "status", "eta", "items"}
    assert expected_keys.issubset(data.keys())
