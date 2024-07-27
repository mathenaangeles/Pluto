from llama_index.core.tools import FunctionTool

def get_market_capitalization(share_price, shares_outstanding):
    return share_price * shares_outstanding
get_market_capitalization_tool = FunctionTool.from_defaults(fn=get_market_capitalization)

def get_diluted_market_capitalization(share_price, shared_authorized):
    return share_price * shared_authorized
get_diluted_market_capitalization_tool = FunctionTool.from_defaults(fn=get_diluted_market_capitalization)

def get_times_revenue(selling_price, annual_revenue):
    return selling_price/annual_revenue
get_times_revenue_tool = FunctionTool.from_defaults(fn=get_times_revenue)

def get_earnings_multiplier(price_per_share, earnings_per_share):
    return price_per_share/earnings_per_share
get_earnings_multiplier_tool = FunctionTool.from_defaults(fn=get_earnings_multiplier)

def get_discounted_cash_flow(cash_flows, discount_rate):
    discounted_cash_flow = 0
    for time, cash_flow in enumerate(cash_flows, start=1):
        discounted_cash_flow += cash_flow / ((1 + discount_rate) ** time)
    return discounted_cash_flow
get_discounted_cash_flow_tool = FunctionTool.from_defaults(fn=get_discounted_cash_flow)

def get_shareholder_equity(assets, liabilities):
    return assets-liabilities
get_shareholder_equity_tool = FunctionTool.from_defaults(fn=get_shareholder_equity)

def get_book_value_per_share(shareholder_equity, preferred_stock, shares_outstanding):
    return (shareholder_equity-preferred_stock)/shares_outstanding
get_book_value_per_share_tool = FunctionTool.from_defaults(fn=get_book_value_per_share)

def get_price_to_book_ratio(market_share_price, book_value_per_share):
    return market_share_price/book_value_per_share
get_price_to_book_ratio_tool = FunctionTool.from_defaults(fn=get_price_to_book_ratio)


