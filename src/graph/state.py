from typing_extensions import Annotated, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage
from colorama import Fore, Style


import json


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


def convert_to_serializable(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)


def _print_json(output):
    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2, ensure_ascii=False))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(output)


def _fmt_num(val, fmt=",.0f", prefix="$", suffix=""):
    """Format a number nicely, return 'N/A' for None."""
    if val is None or val == "None":
        return f"{Fore.RED}N/A{Style.RESET_ALL}"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return f"{Fore.RED}N/A{Style.RESET_ALL}"
    # Auto-scale large numbers
    if prefix == "$" and abs(v) >= 1e12:
        return f"{Fore.CYAN}{prefix}{v/1e12:{fmt.replace(',', '')}}T{suffix}{Style.RESET_ALL}"
    elif prefix == "$" and abs(v) >= 1e9:
        return f"{Fore.CYAN}{prefix}{v/1e9:{fmt.replace(',', '')}}B{suffix}{Style.RESET_ALL}"
    elif prefix == "$" and abs(v) >= 1e6:
        return f"{Fore.CYAN}{prefix}{v/1e6:{fmt.replace(',', '')}}M{suffix}{Style.RESET_ALL}"
    return f"{Fore.CYAN}{prefix}{v:{fmt}}{suffix}{Style.RESET_ALL}"


def _fmt_pct(val, already_ratio=True):
    """Format as percentage. If already_ratio=True, 0.15 -> 15.0%."""
    if val is None or val == "None":
        return f"{Fore.RED}N/A{Style.RESET_ALL}"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return f"{Fore.RED}N/A{Style.RESET_ALL}"
    if already_ratio:
        v = v * 100
    color = Fore.GREEN if v > 0 else (Fore.RED if v < 0 else Fore.YELLOW)
    return f"{color}{v:+.1f}%{Style.RESET_ALL}" if v != 0 else f"{Fore.YELLOW}{v:.1f}%{Style.RESET_ALL}"


def _fmt_ratio(val):
    if val is None or val == "None":
        return f"{Fore.RED}N/A{Style.RESET_ALL}"
    try:
        v = float(val)
        return f"{Fore.CYAN}{v:.2f}{Style.RESET_ALL}"
    except (ValueError, TypeError):
        return f"{Fore.RED}N/A{Style.RESET_ALL}"


def _score_bar(score, max_score, label):
    """Print a visual score bar like: 护城河: ████░░░░░░ 3/5"""
    try:
        s = float(score)
        m = float(max_score) if max_score else 10
    except (ValueError, TypeError):
        s, m = 0, 10
    filled = int(s / m * 10) if m > 0 else 0
    bar = f"{Fore.GREEN}{'█' * filled}{Fore.WHITE}{'░' * (10 - filled)}{Style.RESET_ALL}"
    return f"  {label:<12} {bar} {Fore.YELLOW}{s:.1f}/{m:.0f}{Style.RESET_ALL}"


def _print_section(title, content):
    print(f"\n  {Fore.WHITE}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print(f"  {'─' * 56}")
    print(content)


def _print_key_metrics(metrics_data):
    """Print key financial metrics from fundamental_analysis or raw metrics."""
    if not metrics_data:
        return
    m = metrics_data.get("metrics", metrics_data)
    lines = []

    row1 = (
        f"  ROE: {_fmt_pct(m.get('return_on_equity'))}  |  "
        f"ROA: {_fmt_pct(m.get('return_on_assets'))}  |  "
        f"EPS: {_fmt_num(m.get('earnings_per_share'), '.2f')}"
    )
    row2 = (
        f"  毛利率: {_fmt_pct(m.get('gross_margin'))}  |  "
        f"净利率: {_fmt_pct(m.get('net_margin'))}  |  "
        f"营收增长: {_fmt_pct(m.get('revenue_growth'))}"
    )
    row3 = (
        f"  流动比率: {_fmt_ratio(m.get('current_ratio'))}  |  "
        f"负债/权益: {_fmt_ratio(m.get('debt_to_equity'))}  |  "
        f"负债/资产: {_fmt_pct(m.get('debt_to_assets'))}"
    )
    lines.extend([row1, row2, row3])
    print("\n".join(lines))


def _print_intrinsic_value(iv_data, market_cap, margin_of_safety):
    """Print intrinsic value analysis."""
    if not iv_data:
        return
    iv = iv_data.get("intrinsic_value")
    raw_iv = iv_data.get("raw_intrinsic_value")
    oe = iv_data.get("owner_earnings")
    assumptions = iv_data.get("assumptions", {})

    lines = []
    lines.append(f"  Owner Earnings:   {_fmt_num(oe, '.1f')}")
    lines.append(f"  内在价值 (保守):  {_fmt_num(iv, '.1f')}")
    if raw_iv:
        lines.append(f"  内在价值 (原始):  {_fmt_num(raw_iv, '.1f')}")
    lines.append(f"  当前市值:         {_fmt_num(market_cap, '.1f')}")

    if margin_of_safety is not None and margin_of_safety != "None":
        try:
            mos = float(margin_of_safety)
            mos_color = Fore.GREEN if mos > 0 else Fore.RED
            label = "低估" if mos > 0 else "高估"
            lines.append(f"  安全边际:         {mos_color}{mos:+.1%} ← {label}{Style.RESET_ALL}")
        except (ValueError, TypeError):
            pass

    if assumptions:
        g1 = assumptions.get("stage1_growth")
        g2 = assumptions.get("stage2_growth")
        dr = assumptions.get("discount_rate")
        if g1 is not None:
            lines.append(f"  假设: 第一阶段增长 {float(g1):.1%}, 第二阶段 {float(g2):.1%}, 折现率 {float(dr):.1%}")

    print("\n".join(lines))


def _print_munger_valuation(val_data):
    """Print Munger-style valuation."""
    if not val_data:
        return
    lines = []
    ivr = val_data.get("intrinsic_value_range", {})
    lines.append(f"  FCF 收益率:       {_fmt_pct(val_data.get('fcf_yield'))}")
    lines.append(f"  标准化 FCF:       {_fmt_num(val_data.get('normalized_fcf'), '.1f')}")
    if ivr:
        lines.append(f"  保守估值:         {_fmt_num(ivr.get('conservative'), '.1f')}")
        lines.append(f"  合理估值:         {_fmt_num(ivr.get('reasonable'), '.1f')}")
        lines.append(f"  乐观估值:         {_fmt_num(ivr.get('optimistic'), '.1f')}")
    mos = val_data.get("margin_of_safety_vs_fair_value")
    if mos is not None:
        try:
            mos_v = float(mos)
            mos_color = Fore.GREEN if mos_v > 0 else Fore.RED
            label = "低估" if mos_v > 0 else "高估"
            lines.append(f"  安全边际(vs合理):  {mos_color}{mos_v:+.1%} ← {label}{Style.RESET_ALL}")
        except (ValueError, TypeError):
            pass
    print("\n".join(lines))


def _print_lynch_valuation(val_data):
    """Print Lynch-style valuation (PEG focus)."""
    if not val_data:
        return
    details = val_data.get("details", "")
    # Extract key ratios from details string
    print(f"  {Fore.WHITE}{details}{Style.RESET_ALL}")


def _format_detailed_analysis(ticker, data, agent_name):
    """Format detailed analysis in a human-readable Chinese format."""
    print(f"\n  {Fore.CYAN}{Style.BRIGHT}{'─' * 20} {ticker} 详细分析 {'─' * 20}{Style.RESET_ALL}")

    score = data.get("score")
    max_score = data.get("max_score")
    if score is not None and max_score is not None:
        try:
            s = float(score)
            m = float(max_score)
            pct = s / m * 100 if m > 0 else 0
            color = Fore.GREEN if pct >= 60 else (Fore.YELLOW if pct >= 40 else Fore.RED)
            print(f"\n  {Fore.WHITE}{Style.BRIGHT}综合评分: {color}{s:.1f}/{m:.0f} ({pct:.0f}%){Style.RESET_ALL}")
        except (ValueError, TypeError):
            pass

    # Collect all sub-analyses for score bars
    analysis_keys = {
        "fundamental_analysis": "基本面",
        "consistency_analysis": "一致性",
        "moat_analysis": "护城河",
        "pricing_power_analysis": "定价权",
        "book_value_analysis": "账面价值",
        "management_analysis": "管理层",
        "predictability_analysis": "可预测性",
        "growth_analysis": "增长",
        "valuation_analysis": "估值",
        "fundamentals_analysis": "基本面",
        "sentiment_analysis": "市场情绪",
        "insider_activity": "内幕交易",
    }

    # Print score bars
    bars = []
    for key, label in analysis_keys.items():
        sub = data.get(key)
        if sub and isinstance(sub, dict) and "score" in sub:
            sub_max = sub.get("max_score", 10)
            if isinstance(sub_max, str):
                try:
                    sub_max = float(sub_max)
                except ValueError:
                    sub_max = 10
            bars.append(_score_bar(sub["score"], sub_max, label))

    if bars:
        _print_section("各项评分", "\n".join(bars))

    # Print details for each sub-analysis
    detail_lines = []
    for key, label in analysis_keys.items():
        sub = data.get(key)
        if sub and isinstance(sub, dict) and "details" in sub:
            details_text = sub["details"]
            if details_text and details_text != "Limited pricing power analysis available":
                detail_lines.append(f"  {Fore.YELLOW}{label}:{Style.RESET_ALL} {details_text}")

    if detail_lines:
        _print_section("分析详情", "\n".join(detail_lines))

    # Print key financial metrics if available
    fund = data.get("fundamental_analysis")
    if fund and isinstance(fund, dict) and "metrics" in fund:
        _print_section("关键财务指标", "")
        _print_key_metrics(fund)

    # Print intrinsic value (Warren Buffett style)
    iv = data.get("intrinsic_value_analysis")
    if iv and isinstance(iv, dict) and iv.get("intrinsic_value"):
        _print_section("内在价值 (DCF)", "")
        _print_intrinsic_value(iv, data.get("market_cap"), data.get("margin_of_safety"))

    # Print Munger valuation
    val = data.get("valuation_analysis")
    if val and isinstance(val, dict) and val.get("intrinsic_value_range"):
        _print_section("估值分析 (Munger)", "")
        _print_munger_valuation(val)
    elif val and isinstance(val, dict) and "details" in val and "PEG" in str(val.get("details", "")):
        _print_section("估值分析 (GARP/PEG)", "")
        _print_lynch_valuation(val)

    # News sentiment
    ns = data.get("news_sentiment")
    if ns and ns != "No news data available":
        print(f"\n  {Fore.YELLOW}新闻情绪:{Style.RESET_ALL} {ns}")


def show_agent_reasoning(output, agent_name, detailed_analysis=None):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    if detailed_analysis:
        for ticker, data in detailed_analysis.items():
            if isinstance(data, dict):
                _format_detailed_analysis(ticker, data, agent_name)
            else:
                _print_json(data)

        # Print final signal
        print(f"\n  {Fore.WHITE}{Style.BRIGHT}{'─' * 20} 最终信号 {'─' * 20}{Style.RESET_ALL}")

    _print_json(output)

    print("=" * 48)
