import datetime
import logging
import os
import time
import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
    CompanyFacts,
)

# Global cache instance
_cache = get_cache()


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _pct(val) -> float | None:
    """Convert a percentage value (e.g. 46.9) to a ratio (0.469)."""
    f = _safe_float(val)
    return f / 100.0 if f is not None else None


# ── AkShare line-item name mapping (Chinese → English) ───────────────
_LINE_ITEM_MAP = {
    # Income statement (综合损益表)
    "revenue": "主营收入",
    "net_income": "净利润",
    "gross_profit": "毛利",
    "operating_income": "营业利润",
    "ebit": "持续经营税前利润",
    "ebitda": None,  # not directly available, computed if needed
    "interest_expense": None,
    "earnings_per_share": "基本每股收益-普通股",
    # Balance sheet (资产负债表)
    "total_assets": "总资产",
    "total_liabilities": "总负债",
    "shareholders_equity": "股东权益合计",
    "current_assets": "流动资产合计",
    "current_liabilities": "流动负债合计",
    "cash_and_equivalents": "现金及现金等价物",
    "total_debt": "长期负债",
    "outstanding_shares": "基本加权平均股数-普通股",
    "book_value_per_share": None,  # computed: equity / shares
    # Cash flow statement (现金流量表)
    "free_cash_flow": None,  # computed: operating_cash_flow - capex
    "capital_expenditure": "购买固定资产",
    "depreciation_and_amortization": "折旧及摊销",
    "operating_cash_flow": "经营活动产生的现金流量净额",
    "dividends_and_other_cash_distributions": "股息支付",
    "issuance_or_purchase_of_equity_shares": "回购股份",
    "working_capital": None,  # computed: current_assets - current_liabilities
}

# Reverse map: Chinese name → list of English keys
_CN_TO_EN: dict[str, list[str]] = {}
for _en, _cn in _LINE_ITEM_MAP.items():
    if _cn is not None:
        _CN_TO_EN.setdefault(_cn, []).append(_en)


def _is_a_share(ticker: str) -> bool:
    """A 股 ticker 全是数字 (600036, 000001)，美股是字母 (AAPL)。"""
    return ticker.isdigit()


def _a_share_symbol(ticker: str) -> str:
    """Convert pure-digit A-share ticker to prefixed symbol (sh600036 / sz000001)."""
    return f"sh{ticker}" if ticker.startswith("6") else f"sz{ticker}"


# ── A 股列名映射 (宽格式报表，兼容银行和普通公司) ──────────────
_A_SHARE_LINE_ITEM_MAP: dict[str, list[str]] = {
    # Income statement (利润表)
    "revenue": ["营业收入", "营业总收入"],
    "net_income": ["净利润", "归属于母公司的净利润", "归属于母公司所有者的净利润"],
    "operating_income": ["营业利润"],
    "gross_profit": [],  # computed: revenue - operating_cost
    "ebit": ["利润总额"],
    # Balance sheet (资产负债表)
    "total_assets": ["资产合计", "资产总计"],
    "total_liabilities": ["负债合计"],
    "shareholders_equity": ["归属于母公司股东的权益", "归属于母公司所有者权益合计",
                           "所有者权益（或股东权益）合计", "股东权益合计"],
    "current_assets": ["流动资产合计"],
    "current_liabilities": ["流动负债合计"],
    "cash_and_equivalents": ["货币资金", "现金及存放中央银行款项"],
    "total_debt": ["长期借款"],
    "outstanding_shares": [],  # from stock_individual_info_em
    # Cash flow statement (现金流量表)
    "operating_cash_flow": ["经营活动产生的现金流量净额"],
    "capital_expenditure": ["购建固定资产、无形资产和其他长期资产所支付的现金"],
    "depreciation_and_amortization": ["折旧费", "固定资产折旧、油气资产折耗、生产性生物资产折旧"],
    "dividends_and_other_cash_distributions": ["分配股利、利润或偿付利息所支付的现金"],
    "issuance_or_purchase_of_equity_shares": [],  # not easily available
    # Computed fields
    "free_cash_flow": [],
    "working_capital": [],
    "book_value_per_share": [],
    "earnings_per_share": [],  # from indicator API
}


def _ak_retry(func, max_retries=3):
    """Retry an AkShare call with backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            if any(kw in msg for kw in ("Too Many Requests", "Rate", "Timeout", "timed out", "Connection")):
                delay = 5 + (5 * attempt)
                logger.info("AkShare request failed, retrying in %ds (attempt %d/%d): %s", delay, attempt + 1, max_retries, msg)
                time.sleep(delay)
                continue
            raise
    return func()  # final attempt


# ── Prices ─────────────────────────────────────────────────────────────
def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch daily OHLCV price data via AkShare (sina source)."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    try:
        if _is_a_share(ticker):
            sym = _a_share_symbol(ticker)
            df = _ak_retry(lambda: ak.stock_zh_a_daily(symbol=sym, adjust="qfq"))
            date_col, open_col, close_col, high_col, low_col, vol_col = "date", "open", "close", "high", "low", "volume"
        else:
            df = _ak_retry(lambda: ak.stock_us_daily(symbol=ticker, adjust=""))
            date_col, open_col, close_col, high_col, low_col, vol_col = "date", "open", "close", "high", "low", "volume"
    except Exception as e:
        logger.warning("AkShare price fetch failed for %s: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    # Filter by date range
    df["_date"] = pd.to_datetime(df[date_col])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["_date"] >= start_dt) & (df["_date"] <= end_dt)]

    if df.empty:
        return []

    prices: list[Price] = []
    for _, row in df.iterrows():
        prices.append(
            Price(
                open=float(row[open_col]),
                close=float(row[close_col]),
                high=float(row[high_col]),
                low=float(row[low_col]),
                volume=int(row[vol_col]),
                time=row["_date"].strftime("%Y-%m-%dT%H:%M:%S"),
            )
        )

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ── Financial Metrics ──────────────────────────────────────────────────
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Build FinancialMetrics from AkShare US financial indicators."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    try:
        if _is_a_share(ticker):
            df = _ak_retry(lambda: ak.stock_financial_analysis_indicator(symbol=ticker, start_year="2015"))
        else:
            indicator_type = "年报" if period in ("ttm", "annual") else "单季报"
            df = _ak_retry(lambda: ak.stock_financial_us_analysis_indicator_em(symbol=ticker, indicator=indicator_type))
    except Exception as e:
        logger.warning("AkShare financial metrics fetch failed for %s: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    if _is_a_share(ticker):
        return _build_a_share_metrics(ticker, df, end_date, period, limit, cache_key)
    else:
        return _build_us_metrics(ticker, df, end_date, period, limit, cache_key)


def _build_a_share_metrics(ticker, df, end_date, period, limit, cache_key):
    """Build FinancialMetrics from A-share indicator data (sina source)."""
    # 日期 column is datetime.date objects
    df["_report_date"] = pd.to_datetime(df["日期"])

    # Only keep annual reports (month=12, day=31) if period is annual/ttm
    if period in ("ttm", "annual"):
        df = df[(df["_report_date"].dt.month == 12) & (df["_report_date"].dt.day == 31)]

    df = df[df["_report_date"] <= pd.to_datetime(end_date)]
    df = df.sort_values("_report_date", ascending=False).head(limit)

    results: list[FinancialMetrics] = []
    for _, row in df.iterrows():
        report_date = row["_report_date"].strftime("%Y-%m-%d")
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=report_date,
            period=period,
            currency="CNY",
            market_cap=None,
            enterprise_value=None,
            price_to_earnings_ratio=None,
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=_pct(row.get("销售毛利率(%)")),
            operating_margin=_pct(row.get("营业利润率(%)")),
            net_margin=_pct(row.get("销售净利率(%)")),
            return_on_equity=_pct(row.get("净资产收益率(%)")),
            return_on_assets=_pct(row.get("总资产利润率(%)")),
            return_on_invested_capital=None,
            asset_turnover=_safe_float(row.get("总资产周转率(次)")),
            inventory_turnover=_safe_float(row.get("存货周转率(次)")),
            receivables_turnover=_safe_float(row.get("应收账款周转率(次)")),
            days_sales_outstanding=_safe_float(row.get("应收账款周转天数(天)")),
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=_safe_float(row.get("流动比率")),
            quick_ratio=_safe_float(row.get("速动比率")),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=_pct(row.get("负债与所有者权益比率(%)")),
            debt_to_assets=_pct(row.get("资产负债率(%)")),
            interest_coverage=None,
            revenue_growth=_pct(row.get("主营业务收入增长率(%)")),
            earnings_growth=_pct(row.get("净利润增长率(%)")),
            book_value_growth=_pct(row.get("净资产增长率(%)")),
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=_safe_float(row.get("摊薄每股收益(元)")),
            book_value_per_share=_safe_float(row.get("每股净资产_调整前(元)")),
            free_cash_flow_per_share=None,
        )
        results.append(metrics)

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in results])
    return results[:limit]


def _build_us_metrics(ticker, df, end_date, period, limit, cache_key):
    """Build FinancialMetrics from US indicator data (EM source)."""
    # Filter by end_date
    df["_report_date"] = pd.to_datetime(df["REPORT_DATE"])
    df = df[df["_report_date"] <= pd.to_datetime(end_date)]
    df = df.sort_values("_report_date", ascending=False).head(limit)

    results: list[FinancialMetrics] = []
    for _, row in df.iterrows():
        report_date = row["_report_date"].strftime("%Y-%m-%d")

        market_cap = None

        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=report_date,
            period=period,
            currency=str(row.get("CURRENCY_ABBR", "USD")),
            market_cap=market_cap,
            enterprise_value=None,
            price_to_earnings_ratio=None,
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=_pct(row.get("GROSS_PROFIT_RATIO")),
            operating_margin=None,
            net_margin=_pct(row.get("NET_PROFIT_RATIO")),
            return_on_equity=_pct(row.get("ROE_AVG")),
            return_on_assets=_pct(row.get("ROA")),
            return_on_invested_capital=None,
            asset_turnover=_safe_float(row.get("TOTAL_ASSETS_TR")),
            inventory_turnover=_safe_float(row.get("INVENTORY_TR")),
            receivables_turnover=_safe_float(row.get("ACCOUNTS_RECE_TR")),
            days_sales_outstanding=_safe_float(row.get("ACCOUNTS_RECE_TDAYS")),
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=_safe_float(row.get("CURRENT_RATIO")),
            quick_ratio=_safe_float(row.get("SPEED_RATIO")),
            cash_ratio=None,
            operating_cash_flow_ratio=_safe_float(row.get("OCF_LIQDEBT")),
            debt_to_equity=_safe_float(row.get("EQUITY_RATIO")),
            debt_to_assets=_pct(row.get("DEBT_ASSET_RATIO")),
            interest_coverage=None,
            revenue_growth=_pct(row.get("OPERATE_INCOME_YOY")),
            earnings_growth=_pct(row.get("PARENT_HOLDER_NETPROFIT_YOY")),
            book_value_growth=None,
            earnings_per_share_growth=_pct(row.get("BASIC_EPS_YOY")),
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=_safe_float(row.get("BASIC_EPS")),
            book_value_per_share=None,
            free_cash_flow_per_share=None,
        )
        results.append(metrics)

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in results])
    return results[:limit]


# ── Line Items (Financial Statements) ──────────────────────────────────
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch financial statement line items from AkShare."""
    if _is_a_share(ticker):
        return _search_line_items_a_share(ticker, line_items, end_date, period, limit)
    else:
        return _search_line_items_us(ticker, line_items, end_date, period, limit)


def _search_line_items_us(ticker, line_items, end_date, period, limit):
    """Fetch US stock line items (EM source, long format)."""
    indicator_type = "年报" if period in ("ttm", "annual") else "单季报"

    try:
        income_df = _ak_retry(lambda: ak.stock_financial_us_report_em(stock=ticker, symbol="综合损益表", indicator=indicator_type))
        balance_df = _ak_retry(lambda: ak.stock_financial_us_report_em(stock=ticker, symbol="资产负债表", indicator=indicator_type))
        cashflow_df = _ak_retry(lambda: ak.stock_financial_us_report_em(stock=ticker, symbol="现金流量表", indicator=indicator_type))
    except Exception as e:
        logger.warning("AkShare US financials fetch failed for %s: %s", ticker, e)
        return []

    frames = []
    for df in [income_df, balance_df, cashflow_df]:
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return []

    combined = pd.concat(frames, ignore_index=True)
    combined["_report_date"] = pd.to_datetime(combined["REPORT_DATE"])
    combined = combined[combined["_report_date"] <= pd.to_datetime(end_date)]

    report_dates = sorted(combined["_report_date"].unique(), reverse=True)[:limit]

    results: list[LineItem] = []
    for rd in report_dates:
        report_date_str = pd.Timestamp(rd).strftime("%Y-%m-%d")
        period_df = combined[combined["_report_date"] == rd]

        cn_lookup: dict[str, float | None] = {}
        for _, row in period_df.iterrows():
            cn_name = str(row.get("ITEM_NAME", ""))
            cn_lookup[cn_name] = _safe_float(row.get("AMOUNT"))

        item_data: dict = {
            "ticker": ticker,
            "report_period": report_date_str,
            "period": period if period != "ttm" else "annual",
            "currency": "USD",
        }

        for requested_name in line_items:
            cn_name = _LINE_ITEM_MAP.get(requested_name)
            value = None

            if cn_name and cn_name in cn_lookup:
                value = cn_lookup[cn_name]

            if value is None and requested_name == "free_cash_flow":
                ocf = cn_lookup.get("经营活动产生的现金流量净额")
                capex = cn_lookup.get("购买固定资产")
                if ocf is not None and capex is not None:
                    value = ocf + capex
                elif ocf is not None:
                    value = ocf

            if value is None and requested_name == "working_capital":
                ca = cn_lookup.get("流动资产合计")
                cl = cn_lookup.get("流动负债合计")
                if ca is not None and cl is not None:
                    value = ca - cl

            if value is None and requested_name == "book_value_per_share":
                eq = cn_lookup.get("股东权益合计")
                shares = cn_lookup.get("基本加权平均股数-普通股")
                if eq is not None and shares is not None and shares != 0:
                    value = eq / shares

            item_data[requested_name] = value

        results.append(LineItem(**item_data))

    return results[:limit]


def _lookup_a_share_col(row, col_names: list[str]) -> float | None:
    """Try multiple column names and return the first non-null value."""
    for col in col_names:
        if col in row.index:
            val = _safe_float(row[col])
            if val is not None:
                return val
    return None


def _search_line_items_a_share(ticker, line_items, end_date, period, limit):
    """Fetch A-share line items (sina source, wide format)."""
    try:
        income_df = _ak_retry(lambda: ak.stock_financial_report_sina(stock=ticker, symbol="利润表"))
        balance_df = _ak_retry(lambda: ak.stock_financial_report_sina(stock=ticker, symbol="资产负债表"))
        cashflow_df = _ak_retry(lambda: ak.stock_financial_report_sina(stock=ticker, symbol="现金流量表"))
    except Exception as e:
        logger.warning("AkShare A-share financials fetch failed for %s: %s", ticker, e)
        return []

    # Merge all three DataFrames on 报告日
    dfs = []
    for df in [income_df, balance_df, cashflow_df]:
        if df is not None and not df.empty:
            dfs.append(df.set_index("报告日"))
    if not dfs:
        return []

    # Join on report date index (outer join to keep all dates)
    combined = dfs[0]
    for df in dfs[1:]:
        # Drop duplicate columns to avoid _x/_y suffixes
        overlap_cols = combined.columns.intersection(df.columns)
        combined = combined.join(df.drop(columns=overlap_cols), how="outer")

    # Convert index to datetime for filtering
    combined["_report_date"] = pd.to_datetime(combined.index, format="%Y%m%d")

    # Only keep annual reports if period is annual/ttm
    if period in ("ttm", "annual"):
        combined = combined[combined.index.astype(str).str.endswith("1231")]

    combined = combined[combined["_report_date"] <= pd.to_datetime(end_date)]
    combined = combined.sort_values("_report_date", ascending=False).head(limit)

    # Get outstanding_shares: try stock_individual_info_em, fallback to indicator data
    outstanding_shares = None
    try:
        info_df = _ak_retry(lambda: ak.stock_individual_info_em(symbol=ticker))
        for _, r in info_df.iterrows():
            if r["item"] == "总股本":
                outstanding_shares = _safe_float(r["value"])
                break
    except Exception:
        pass
    # Fallback: compute from indicator (equity / book_value_per_share)
    if outstanding_shares is None:
        try:
            ind_df = _ak_retry(lambda: ak.stock_financial_analysis_indicator(symbol=ticker, start_year="2020"))
            if ind_df is not None and not ind_df.empty:
                r = ind_df.iloc[-1]  # latest
                bvps = _safe_float(r.get("每股净资产_调整前(元)"))
                equity_pct = _safe_float(r.get("股东权益比率(%)"))
                total_assets_val = _safe_float(r.get("总资产(元)"))
                if bvps and bvps != 0 and equity_pct and total_assets_val:
                    equity = total_assets_val * equity_pct / 100
                    outstanding_shares = equity / bvps
        except Exception:
            pass

    results: list[LineItem] = []
    for _, row in combined.iterrows():
        report_date_str = row["_report_date"].strftime("%Y-%m-%d")

        item_data: dict = {
            "ticker": ticker,
            "report_period": report_date_str,
            "period": period if period != "ttm" else "annual",
            "currency": "CNY",
        }

        for requested_name in line_items:
            col_names = _A_SHARE_LINE_ITEM_MAP.get(requested_name, [])
            value = _lookup_a_share_col(row, col_names) if col_names else None

            # Handle capital_expenditure (make it negative as convention)
            if requested_name == "capital_expenditure" and value is not None:
                value = -abs(value)

            # Handle dividends (make it negative as convention)
            if requested_name == "dividends_and_other_cash_distributions" and value is not None:
                value = -abs(value)

            # Computed: free_cash_flow = operating_cash_flow + capital_expenditure
            if value is None and requested_name == "free_cash_flow":
                ocf = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("operating_cash_flow", []))
                capex_raw = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("capital_expenditure", []))
                if ocf is not None and capex_raw is not None:
                    value = ocf - abs(capex_raw)
                elif ocf is not None:
                    value = ocf

            # Computed: working_capital
            if value is None and requested_name == "working_capital":
                ca = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("current_assets", []))
                cl = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("current_liabilities", []))
                if ca is not None and cl is not None:
                    value = ca - cl

            # Computed: gross_profit = revenue - operating_cost
            if value is None and requested_name == "gross_profit":
                rev = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("revenue", []))
                cost = _lookup_a_share_col(row, ["营业成本", "营业总成本"])
                if rev is not None and cost is not None:
                    value = rev - cost

            # outstanding_shares from individual_info
            if requested_name == "outstanding_shares":
                value = outstanding_shares

            # book_value_per_share
            if value is None and requested_name == "book_value_per_share":
                eq = _lookup_a_share_col(row, _A_SHARE_LINE_ITEM_MAP.get("shareholders_equity", []))
                if eq is not None and outstanding_shares is not None and outstanding_shares > 0:
                    value = eq / outstanding_shares

            item_data[requested_name] = value

        results.append(LineItem(**item_data))

    return results[:limit]


# ── Insider Trades ─────────────────────────────────────────────────────
def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """AkShare does not provide US insider trade data. Returns empty list."""
    return []


# ── Company News ───────────────────────────────────────────────────────
def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news via AkShare (eastmoney source)."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    try:
        df = _ak_retry(lambda: ak.stock_news_em(symbol=ticker))
    except Exception as e:
        logger.warning("AkShare news fetch failed for %s: %s", ticker, e)
        return []

    if df is None or df.empty:
        return []

    results: list[CompanyNews] = []
    for _, row in df.iterrows():
        pub_date = str(row.get("发布时间", ""))
        date_str = pub_date[:10]

        if date_str and date_str > end_date:
            continue
        if start_date and date_str and date_str < start_date:
            continue

        title = str(row.get("新闻标题", ""))
        source = str(row.get("文章来源", "Unknown"))
        url = str(row.get("新闻链接", ""))

        results.append(
            CompanyNews(
                ticker=ticker,
                title=title,
                author=None,
                source=source,
                date=pub_date[:19] if pub_date else end_date,
                url=url,
                sentiment=None,
            )
        )

        if len(results) >= limit:
            break

    _cache.set_company_news(cache_key, [n.model_dump() for n in results])
    return results


# ── Market Cap ─────────────────────────────────────────────────────────
def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Estimate market cap."""
    try:
        if _is_a_share(ticker):
            # A-share: try stock_individual_info_em first, fallback to price × shares
            try:
                info_df = _ak_retry(lambda: ak.stock_individual_info_em(symbol=ticker))
                if info_df is not None and not info_df.empty:
                    for _, row in info_df.iterrows():
                        if row["item"] == "总市值":
                            val = _safe_float(row["value"])
                            if val is not None:
                                return val
            except Exception:
                pass

            # Fallback: price × shares from indicator
            try:
                sym = _a_share_symbol(ticker)
                price_df = _ak_retry(lambda: ak.stock_zh_a_daily(symbol=sym, adjust="qfq"))
                if price_df is not None and not price_df.empty:
                    latest_price = float(price_df.iloc[-1]["close"])
                    ind_df = _ak_retry(lambda: ak.stock_financial_analysis_indicator(symbol=ticker, start_year="2020"))
                    if ind_df is not None and not ind_df.empty:
                        for _, r in ind_df.iterrows():
                            total_assets = _safe_float(r.get("总资产(元)"))
                            eps = _safe_float(r.get("摊薄每股收益(元)"))
                            ni = _safe_float(r.get("主营业务利润(元)"))
                            # Get shares from net_income / EPS
                            # Use 归属于母公司的净利润 if available
                            if eps and eps != 0 and total_assets:
                                # Rough estimate from total_assets / 每股净资产
                                bvps = _safe_float(r.get("每股净资产_调整前(元)"))
                                if bvps and bvps != 0:
                                    shares_est = total_assets * _safe_float(r.get("股东权益比率(%)")) / 100 / bvps if _safe_float(r.get("股东权益比率(%)")) else None
                                    if shares_est:
                                        return latest_price * shares_est
                            break
            except Exception:
                pass
            return None
        else:
            # US stock: estimate from price * shares
            df = _ak_retry(lambda: ak.stock_us_daily(symbol=ticker, adjust=""))
            if df is None or df.empty:
                return None
            latest_price = float(df.iloc[-1]["close"])

            ind_df = _ak_retry(lambda: ak.stock_financial_us_analysis_indicator_em(symbol=ticker, indicator="年报"))
            if ind_df is not None and not ind_df.empty:
                row = ind_df.iloc[0]
                net_income = _safe_float(row.get("PARENT_HOLDER_NETPROFIT"))
                eps = _safe_float(row.get("BASIC_EPS"))
                if net_income and eps and eps != 0:
                    shares = net_income / eps
                    return latest_price * shares

            return None
    except Exception as e:
        logger.warning("AkShare market cap fetch failed for %s: %s", ticker, e)
        return None


# ── DataFrame helpers (unchanged interface) ────────────────────────────
def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
