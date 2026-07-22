import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger("agent.tools")

ORDER_DB = {
    "ORD-001": {"order_id": "ORD-001", "status": "Out for delivery", "eta": "15 minutes", "items": "2x Burger, 1x Fries"},
    "ORD-002": {"order_id": "ORD-002", "status": "Preparing", "eta": "25 minutes", "items": "1x Pizza, 1x Drink"},
    "ORD-003": {"order_id": "ORD-003", "status": "Delivered", "eta": "0 minutes", "items": "1x Salad, 2x Wraps"},
    "12345": {"order_id": "12345", "status": "Out for delivery", "eta": "15 minutes", "items": "1x Chicken Burger, 1x Milkshake"},
}


@tool
def get_order_status(order_id: str) -> dict:
    """Call this tool to retrieve the current delivery status of a customer's order. Requires a valid order_id (e.g., 'ORD-001' or '12345'). Always call this before reporting order status to a user."""
    logger.info("Tool invoked: get_order_status(order_id='%s')", order_id)

    if not order_id or not order_id.strip():
        logger.error("Tool get_order_status failed: ValueError - Empty order_id")
        return {"error": "Order ID cannot be empty", "code": 400}

    order = ORDER_DB.get(order_id.strip())
    if order is None:
        logger.warning("Tool get_order_status: order_id='%s' not found", order_id)
        return {"error": f"Order '{order_id}' not found", "code": 404}

    logger.info("Tool result: %s", order)
    return order
