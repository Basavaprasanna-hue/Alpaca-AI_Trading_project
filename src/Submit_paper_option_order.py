
#This file contains the code to submit the paper order once it passes the live options monitor resolution

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

API_KEY = 
API_SECRET = 

OPTION_SYMBOL = ""  # Copy exact PASS=True symbol
LIMIT_PRICE =                   # Your maximum price per share
QUANTITY = 

client = TradingClient(
    API_KEY,
    API_SECRET,
    paper=True,
)

account = client.get_account()

options_buying_power = getattr(
    account,
    "options_buying_power",
    account.buying_power,
)

estimated_debit = LIMIT_PRICE * 100 * QUANTITY

print("Paper account:", True)
print("Symbol:", OPTION_SYMBOL)
print("Side: BUY")
print("Quantity:", QUANTITY)
print("Limit price:", LIMIT_PRICE)
print("Maximum debit:", estimated_debit)
print("Options buying power:", options_buying_power)

if float(options_buying_power) < estimated_debit:
    raise RuntimeError("Insufficient options buying power; order not submitted.")

order_request = LimitOrderRequest(
    symbol=OPTION_SYMBOL,
    qty=QUANTITY,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    limit_price=LIMIT_PRICE,
)

order = client.submit_order(order_data=order_request)

print("Submitted paper order")
print("Order ID:", order.id)
print("Status:", order.status)
print("Submitted at:", order.submitted_at)
