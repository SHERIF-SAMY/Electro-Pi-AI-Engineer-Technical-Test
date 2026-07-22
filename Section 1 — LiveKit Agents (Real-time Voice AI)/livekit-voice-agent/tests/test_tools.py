import json
import pytest

from agent.tools import ORDER_DB, get_order_status





@pytest.mark.asyncio
async def test_get_order_status_known():
    data = await get_order_status.ainvoke({"order_id": "ORD-001"})
    assert data["order_id"] == "ORD-001"
    assert data["status"] == "Out for delivery"
    assert "eta" in data


@pytest.mark.asyncio
async def test_get_order_status_numeric_id():
    data = await get_order_status.ainvoke({"order_id": "12345"})
    assert data["order_id"] == "12345"
    assert data["status"] == "Out for delivery"


@pytest.mark.asyncio
async def test_get_order_status_unknown_id():
    data = await get_order_status.ainvoke({"order_id": "UNKNOWN-999"})
    assert data["code"] == 404
    assert "error" in data


@pytest.mark.asyncio
async def test_get_order_status_empty_id():
    data = await get_order_status.ainvoke({"order_id": ""})
    assert data["code"] == 400


@pytest.mark.asyncio
async def test_get_order_status_whitespace_id():
    data = await get_order_status.ainvoke({"order_id": "   "})
    assert data["code"] == 400


@pytest.mark.asyncio
async def test_get_order_status_json_serializable():
    data = await get_order_status.ainvoke({"order_id": "ORD-002"})
    assert isinstance(data, dict)
    assert json.dumps(data)


@pytest.mark.asyncio
async def test_get_order_status_all_known_ids():
    for order_id in ORDER_DB:
        data = await get_order_status.ainvoke({"order_id": order_id})
        assert data["order_id"] == order_id
        assert "status" in data
