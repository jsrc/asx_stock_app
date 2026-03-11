import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import ta
import yfinance as yf

DEFAULT_TICKERS = [
    "CBA.AX", "BHP.AX", "CSL.AX", "NAB.AX", "WBC.AX",
    "ANZ.AX", "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX",
    "FMG.AX", "RIO.AX", "COL.AX", "ALL.AX", "SUN.AX",
]

PERIOD_OPTIONS = {
    "1周": "5d",
    "1个月": "1mo",
    "3个月": "3mo",
    "6个月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "3年": "3y",
    "5年": "5y",
}

INDEX_POOL_URLS = {
    "ASX100": "https://fnarena.com/index/ASX100/",
    "ASX200": "https://fnarena.com/index/ASX200/",
    # ASX500 is mapped via ALL-ORDS public constituents (around 500, changes over time).
    "ASX500": "https://fnarena.com/index/ALL-ORDS/",
}

HISTORY_CACHE_DIR = Path(".cache") / "history"
HISTORY_CACHE_MAX_AGE = {
    "5d": 300,
    "1mo": 900,
    "3mo": 1800,
    "6mo": 3600,
    "1y": 14400,
    "2y": 28800,
    "3y": 43200,
    "5y": 86400,
}
DEFAULT_CACHE_MAX_AGE = 43200

def to_1d_series(values):
    if values is None:
        return None
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return None
        return values.iloc[:, 0]
    return values


def normalize_ticker(raw):
    ticker = str(raw).strip().upper()
    if not ticker:
        return ""
    if "." not in ticker:
        ticker = f"{ticker}.AX"
    return ticker


def parse_tickers(raw_text):
    cleaned = (
        raw_text.replace("\n", ",")
        .replace("，", ",")
        .replace("、", ",")
        .replace(";", ",")
        .replace("；", ",")
    )
    items = [normalize_ticker(x) for x in cleaned.split(",")]
    deduped = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def merge_unique_tickers(*ticker_lists):
    merged = []
    seen = set()
    for tickers in ticker_lists:
        for ticker in tickers:
            t = normalize_ticker(ticker)
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
    return merged


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name):
    url = INDEX_POOL_URLS[index_name]
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    symbols = re.findall(
        r"/stock-price-analysis/[^/]+/([A-Z0-9]{2,5})/?",
        response.text,
    )
    return merge_unique_tickers(symbols)


def _cache_file_path(ticker, period):
    safe_ticker = re.sub(r"[^A-Z0-9._-]", "_", normalize_ticker(ticker))
    safe_period = re.sub(r"[^A-Z0-9._-]", "_", str(period).lower())
    return HISTORY_CACHE_DIR / f"{safe_ticker}_{safe_period}.pkl"


def _cache_max_age(period):
    return HISTORY_CACHE_MAX_AGE.get(period, DEFAULT_CACHE_MAX_AGE)


def _is_cache_fresh(cache_file, max_age_seconds):
    if not cache_file.exists():
        return False
    try:
        return (time.time() - cache_file.stat().st_mtime) <= max_age_seconds
    except OSError:
        return False


def _load_cached_history(cache_file):
    if not cache_file.exists():
        return None
    try:
        data = pd.read_pickle(cache_file)
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data.sort_index()
    except Exception:
        return None
    return None


def _save_cached_history(cache_file, data):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(cache_file)
    except Exception:
        pass


def clear_history_cache():
    if not HISTORY_CACHE_DIR.exists():
        return 0
    removed = 0
    for file in HISTORY_CACHE_DIR.glob("*.pkl"):
        try:
            file.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def fetch_history(ticker, period):
    cache_file = _cache_file_path(ticker, period)
    max_age_seconds = _cache_max_age(period)
    cached = _load_cached_history(cache_file)

    if cached is not None and _is_cache_fresh(cache_file, max_age_seconds):
        return cached

    try:
        latest = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        latest = pd.DataFrame()

    if isinstance(latest, pd.DataFrame) and not latest.empty:
        _save_cached_history(cache_file, latest)
        return latest

    if cached is not None:
        return cached

    return pd.DataFrame()


def classify_score(score, trend_ok, momentum_ok):
    trend_gate_ok = trend_ok is not False
    momentum_gate_ok = momentum_ok is not False
    if score >= 75 and trend_gate_ok and momentum_gate_ok:
        return "可考虑买入"
    if score >= 55:
        return "加入观察"
    return "暂不考虑"


def status_to_text(flag):
    if flag is True:
        return "通过 ✅"
    if flag is False:
        return "未通过 ❌"
    return "数据不足 ⏳"


def to_tri_bool(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return bool(value)


def analyze_stock(ticker, period, cfg):
    try:
        data = fetch_history(ticker, period)
        if data.empty:
            return None

        close = to_1d_series(data.get("Close"))
        high = to_1d_series(data.get("High"))
        low = to_1d_series(data.get("Low"))

        if close is None or close.empty:
            return None

        prices = pd.DataFrame(
            {
                "Close": close,
                "High": high if high is not None else close,
                "Low": low if low is not None else close,
            }
        ).dropna()

        close = prices["Close"]
        high = prices["High"]
        low = prices["Low"]

        if len(close) < 15:
            return None

        latest_close = close.iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close) >= 50 else None
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(close) >= 200 else None
        rsi_value = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1] if len(close) >= 14 else None
        adx_value = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1] if len(close) >= 14 else None

        ret_6m = close.pct_change(126).iloc[-1] if len(close) >= 127 else None
        ret_12m = close.pct_change(252).iloc[-1] if len(close) >= 253 else None
        high_20_prev = close.shift(1).rolling(20).max().iloc[-1] if len(close) >= 21 else None
        high_52w = close.rolling(252).max().iloc[-1] if len(close) >= 252 else None

        recent = close.tail(min(252, len(close)))
        drawdown_1y = (recent / recent.cummax() - 1.0).min()

        trend_ok = None
        if ema50 is not None and ema200 is not None:
            trend_ok = to_tri_bool(latest_close > ema50 and ema50 > ema200)

        momentum_ok = None
        if ret_6m is not None and ret_12m is not None:
            momentum_ok = to_tri_bool(ret_6m > 0 and ret_12m > 0)
        elif ret_6m is not None:
            momentum_ok = to_tri_bool(ret_6m > 0)
        elif ret_12m is not None:
            momentum_ok = to_tri_bool(ret_12m > 0)

        breakout_ok = to_tri_bool(latest_close > high_20_prev) if high_20_prev is not None else None
        rsi_ok = to_tri_bool(cfg["rsi_low"] <= rsi_value <= cfg["rsi_high"]) if rsi_value is not None else None
        adx_ok = to_tri_bool(adx_value >= cfg["adx_min"]) if adx_value is not None else None

        signals = {
            "Trend": (trend_ok, cfg["w_trend"]),
            "Momentum": (momentum_ok, cfg["w_momentum"]),
            "Breakout": (breakout_ok, cfg["w_breakout"]),
            "RSI Zone": (rsi_ok, cfg["w_rsi"]),
            "ADX Strength": (adx_ok, cfg["w_adx"]),
        }

        total_weight = sum(weight for flag, weight in signals.values() if flag is not None)
        if total_weight == 0:
            return None
        raw_score = sum(weight for flag, weight in signals.values() if flag is True)
        score = round(100.0 * raw_score / total_weight, 1)

        reasons = []
        reasons.append("趋势向上（收盘价 > EMA50 > EMA200）" if trend_ok is True else ("趋势未确认" if trend_ok is False else "趋势信号数据不足"))
        reasons.append("中期动量为正" if momentum_ok is True else ("中期动量偏弱" if momentum_ok is False else "动量信号数据不足"))
        reasons.append("出现20日突破" if breakout_ok is True else ("未出现新突破" if breakout_ok is False else "突破信号数据不足"))
        reasons.append("RSI处于可参与区间" if rsi_ok is True else ("RSI不在理想区间" if rsi_ok is False else "RSI信号数据不足"))
        reasons.append("ADX显示趋势强度充足" if adx_ok is True else ("ADX趋势强度偏弱" if adx_ok is False else "ADX信号数据不足"))

        return {
            "Ticker": ticker,
            "Score": score,
            "Rating": classify_score(score, trend_ok, momentum_ok),
            "Price": round(float(latest_close), 2),
            "EMA50": round(float(ema50), 2) if ema50 is not None else None,
            "EMA200": round(float(ema200), 2) if ema200 is not None else None,
            "RSI": round(float(rsi_value), 1) if rsi_value is not None else None,
            "ADX": round(float(adx_value), 1) if adx_value is not None else None,
            "Ret6M%": round(float(ret_6m * 100), 1) if ret_6m is not None else None,
            "Ret12M%": round(float(ret_12m * 100), 1) if ret_12m is not None else None,
            "Dist52WHigh%": round(float((latest_close / high_52w - 1.0) * 100), 1) if high_52w not in (None, 0) else None,
            "MaxDD1Y%": round(float(drawdown_1y * 100), 1),
            "Trend": trend_ok,
            "Momentum": momentum_ok,
            "Breakout": breakout_ok,
            "RSIZone": rsi_ok,
            "ADXTrend": adx_ok,
            "Why": " | ".join(reasons),
        }
    except Exception:
        return None


def run_screener(tickers, period, cfg):
    rows = []
    for ticker in tickers:
        result = analyze_stock(ticker, period, cfg)
        if result is not None:
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["Score", "Ret12M%"], ascending=[False, False])


def signal_table(row):
    return pd.DataFrame(
        {
            "策略信号": [
                "趋势方向（收盘价 > EMA50 > EMA200）",
                "动量（6M、12M收益均为正）",
                "突破（突破前20日高点）",
                "RSI 区间过滤",
                "ADX 趋势强度",
            ],
            "结果": [
                status_to_text(row["Trend"]),
                status_to_text(row["Momentum"]),
                status_to_text(row["Breakout"]),
                status_to_text(row["RSIZone"]),
                status_to_text(row["ADXTrend"]),
            ],
        }
    )


def render_indicator_help():
    st.markdown(
        """
- `综合评分`：0~100，按你设置的策略权重计算，分数越高代表与趋势策略越一致。
- `评级`：
  - 可考虑买入：分数较高且趋势、动量都通过
  - 加入观察：有部分信号符合，但不够全面
  - 暂不考虑：当前不符合主要趋势条件
- `6M收益% / 12M收益%`：近6/12个月涨跌幅。
- `RSI`：相对强弱指标，常见理解是 30 以下偏弱，70 以上偏热。
- `ADX`：趋势强度指标，只看强度不看方向，数值越高趋势越明显。
- `距52周高点%`：当前价格离过去52周最高价的百分比，越接近0表示越靠近新高。
- `1年最大回撤%`：过去1年内从阶段高点到低点的最大跌幅（负数越大，波动风险越高）。
        """
    )


st.set_page_config(page_title="ASX 趋势投资筛选器", layout="wide")

st.title("ASX 趋势投资筛选器")
st.caption("仅供个人研究参考，不构成投资建议。")

with st.sidebar:
    st.header("扫描设置")
    st.caption(f"本地缓存目录：{HISTORY_CACHE_DIR.as_posix()}")
    if st.button("清理本地缓存", help="删除已保存的历史行情缓存，下次将重新拉取。"):
        removed_count = clear_history_cache()
        st.success(f"已清理 {removed_count} 个缓存文件。")

    st.subheader("指数股票池")
    use_asx100 = st.checkbox("ASX100（全部成分股）", value=False)
    use_asx200 = st.checkbox("ASX200（全部成分股）", value=False)
    use_asx500 = st.checkbox(
        "ASX500（全部成分股）",
        value=False,
        help="通过 ALL-ORDS 成分股映射，数量随指数调整动态变化。",
    )

    selected_indices = []
    if use_asx100:
        selected_indices.append("ASX100")
    if use_asx200:
        selected_indices.append("ASX200")
    if use_asx500:
        selected_indices.append("ASX500")

    st.subheader("手动补充股票")
    ticker_input = st.text_area(
        "关注股票（逗号或换行分隔）",
        value=", ".join(DEFAULT_TICKERS),
        height=140,
        help="支持输入 CBA 或 CBA.AX。若不带后缀，会自动补为 .AX。可与指数股票池叠加。",
    )
    manual_tickers = parse_tickers(ticker_input)

    index_tickers = []
    failed_indices = []
    for index_name in selected_indices:
        try:
            current = fetch_index_constituents(index_name)
            if current:
                index_tickers = merge_unique_tickers(index_tickers, current)
            else:
                failed_indices.append(index_name)
        except Exception:
            failed_indices.append(index_name)

    tickers = merge_unique_tickers(index_tickers, manual_tickers)

    if selected_indices:
        st.caption(f"已勾选指数：{', '.join(selected_indices)}")
        st.caption(f"指数股票池加载数量：{len(index_tickers)}")
    if failed_indices:
        st.warning(f"以下指数成分股加载失败：{', '.join(failed_indices)}")
    st.caption(f"最终关注股票总数（去重后）：{len(tickers)}")

    period_label = st.selectbox(
        "历史区间",
        list(PERIOD_OPTIONS.keys()),
        index=4,
        help="从1周到5年可选。短周期下，部分长周期指标会自动标记为数据不足。",
    )
    period = PERIOD_OPTIONS[period_label]
    top_n = st.slider("显示前N只股票", min_value=5, max_value=50, value=15)

    st.subheader("阈值设置")
    rsi_low, rsi_high = st.slider(
        "RSI 参与区间",
        min_value=30,
        max_value=75,
        value=(45, 65),
        help="仅当 RSI 落在该区间时，RSI信号才计分。",
    )
    adx_min = st.slider(
        "最小 ADX",
        min_value=10,
        max_value=40,
        value=20,
        help="ADX 越高代表趋势越明显，低于该值则 ADX 信号不计分。",
    )

    st.subheader("策略权重")
    w_trend = st.slider("趋势（EMA）", min_value=1, max_value=5, value=3)
    w_momentum = st.slider("动量（6M/12M）", min_value=1, max_value=5, value=3)
    w_breakout = st.slider("突破（20日）", min_value=1, max_value=5, value=2)
    w_rsi = st.slider("RSI 区间", min_value=1, max_value=5, value=1)
    w_adx = st.slider("ADX 强度", min_value=1, max_value=5, value=1)

cfg = {
    "rsi_low": rsi_low,
    "rsi_high": rsi_high,
    "adx_min": adx_min,
    "w_trend": w_trend,
    "w_momentum": w_momentum,
    "w_breakout": w_breakout,
    "w_rsi": w_rsi,
    "w_adx": w_adx,
}

if not tickers:
    st.error("请至少输入一只股票代码，或勾选至少一个指数股票池。")
    st.stop()

with st.spinner("正在运行多策略扫描..."):
    results = run_screener(tickers, period, cfg)

if results.empty:
    st.warning("未获取到有效股票数据，请检查代码或稍后重试。")
    st.stop()

buy_count = int((results["Rating"] == "可考虑买入").sum())
watch_count = int((results["Rating"] == "加入观察").sum())

c1, c2, c3 = st.columns(3)
c1.metric("已扫描股票", len(results))
c2.metric("可考虑买入", buy_count)
c3.metric("加入观察", watch_count)

if len(tickers) >= 200:
    st.info("当前股票池较大，首次扫描可能需要更长时间。")

if period in {"5d", "1mo", "3mo", "6mo"}:
    st.info("当前为较短历史区间，部分长周期指标（如EMA200、12M收益）可能显示为“数据不足”。")

if hasattr(st, "popover"):
    with st.popover("指标解释（点击查看）"):
        render_indicator_help()
else:
    with st.expander("指标解释（展开查看）"):
        render_indicator_help()

st.subheader("综合排名")

display_cols = [
    "Ticker",
    "Score",
    "Rating",
    "Price",
    "Ret6M%",
    "Ret12M%",
    "RSI",
    "ADX",
    "Dist52WHigh%",
    "MaxDD1Y%",
]

display_df = results[display_cols].rename(
    columns={
        "Ticker": "代码",
        "Score": "综合评分",
        "Rating": "评级",
        "Price": "现价",
        "Ret6M%": "6M收益%",
        "Ret12M%": "12M收益%",
        "RSI": "RSI",
        "ADX": "ADX",
        "Dist52WHigh%": "距52周高点%",
        "MaxDD1Y%": "1年最大回撤%",
    }
)

st.dataframe(
    display_df.head(top_n),
    width="stretch",
    hide_index=True,
    column_config={
        "代码": st.column_config.TextColumn("代码", help="澳股代码，默认以 .AX 结尾。"),
        "综合评分": st.column_config.ProgressColumn(
            "综合评分",
            min_value=0,
            max_value=100,
            format="%.1f",
            help="按多策略加权得到的总分。",
        ),
        "评级": st.column_config.TextColumn("评级", help="根据综合评分和关键趋势/动量条件生成。"),
        "现价": st.column_config.NumberColumn("现价", format="%.2f"),
        "6M收益%": st.column_config.NumberColumn("6M收益%", format="%.1f%%", help="近6个月涨跌幅。"),
        "12M收益%": st.column_config.NumberColumn("12M收益%", format="%.1f%%", help="近12个月涨跌幅。"),
        "RSI": st.column_config.NumberColumn("RSI", format="%.1f", help="相对强弱指标。"),
        "ADX": st.column_config.NumberColumn("ADX", format="%.1f", help="趋势强度指标。"),
        "距52周高点%": st.column_config.NumberColumn("距52周高点%", format="%.1f%%"),
        "1年最大回撤%": st.column_config.NumberColumn("1年最大回撤%", format="%.1f%%"),
    },
)

csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载结果（CSV）",
    data=csv_data,
    file_name="asx_trend_scan_zh.csv",
    mime="text/csv",
)

selected_ticker = st.selectbox("查看单只股票详情", results["Ticker"])
selected_row = results[results["Ticker"] == selected_ticker].iloc[0]

st.markdown(f"**{selected_ticker} | 评分 {selected_row['Score']} | {selected_row['Rating']}**")
st.write("信号解释：", selected_row["Why"])

history = fetch_history(selected_ticker, period)
close = to_1d_series(history.get("Close"))
if close is not None and not close.empty:
    chart = pd.DataFrame(
        {
            "收盘价": close,
            "EMA50": close.ewm(span=50, adjust=False).mean(),
            "EMA200": close.ewm(span=200, adjust=False).mean(),
        }
    ).tail(260)
    st.line_chart(chart)

st.subheader("策略信号检查")
st.dataframe(signal_table(selected_row), width="content", hide_index=True)

