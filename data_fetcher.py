import akshare as ak
import pandas as pd
import requests
import time
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class DataFetcher:
    """
    ETF 数据抓取工具类，提供 Akshare 主路 + 新浪/腾讯备用路的容灾机制。
    支持获取实时行情和历史行情数据。
    """
    
    @staticmethod
    def get_realtime_spot(symbols: List[str]) -> Optional[pd.DataFrame]:
        """
        获取 ETF 实时行情数据，带重试和备用接口切换
        :param symbols: ETF代码列表 (如 ['510300', '159209'])
        :return: DataFrame 包含列: 代码, 名称, 最新价, 涨跌幅, 昨收, volume, turnover, high, low
        """
        max_retries = 3
        retry_delays = [1, 2, 3]

        for attempt in range(max_retries):
            try:
                df = ak.fund_etf_spot_em()
                # 统一列名
                col_map = {
                    '最高价': 'high', '最高': 'high',
                    '最低价': 'low', '最低': 'low',
                    '成交量': 'volume', '成交额': 'turnover',
                    '昨收盘': '昨收', '前收盘': '昨收'
                }
                for old_col, new_col in col_map.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df[new_col] = df[old_col]
                
                if '更新时间' not in df.columns:
                    df['更新时间'] = '实时'
                
                # 过滤出需要的 symbols
                if symbols:
                    df = df[df['代码'].isin(symbols)].copy()
                
                if not df.empty:
                    return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt])
                else:
                    print(f"⚠️ Akshare 获取实时行情失败（已重试{max_retries}次），尝试备用方案...")
                    
                    # 备用方案1: 新浪财经
                    sina_df = DataFetcher._get_sina_spot_data(symbols)
                    if sina_df is not None and not sina_df.empty:
                        print("✅ 成功使用备用方案1 (新浪财经) 获取实时数据")
                        return sina_df
                        
                    # 备用方案2: 腾讯财经
                    tencent_df = DataFetcher._get_tencent_spot_data(symbols)
                    if tencent_df is not None and not tencent_df.empty:
                        print("✅ 成功使用备用方案2 (腾讯财经) 获取实时数据")
                        return tencent_df
                        
        return None

    @staticmethod
    def _format_symbols(symbols: List[str], provider: str = 'sina') -> List[str]:
        formatted = []
        for sym in symbols:
            # 简单判断: 5开头的沪市, 1开头的深市
            if sym.startswith('5') or sym.startswith('6'):
                formatted.append(f"sh{sym}")
            else:
                formatted.append(f"sz{sym}")
        return formatted

    @staticmethod
    def _get_sina_spot_data(symbols: List[str]) -> Optional[pd.DataFrame]:
        try:
            if not symbols:
                return None
            
            req_symbols = DataFetcher._format_symbols(symbols)
            url = f"http://hq.sinajs.cn/list={','.join(req_symbols)}"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                return None

            data_list = []
            for line in resp.text.strip().split('\n'):
                if not line or '="' not in line:
                    continue
                try:
                    code_part = line.split('=')[0]
                    symbol_code = code_part.split('_')[-1][2:]
                    
                    data_str = line.split('="')[1].strip('";')
                    if not data_str:
                        continue
                        
                    fields = data_str.split(',')
                    if len(fields) < 4:
                        continue

                    name = fields[0]
                    pre_close = float(fields[2])
                    price = float(fields[3])
                    high = float(fields[4]) if len(fields) > 4 else price
                    low = float(fields[5]) if len(fields) > 5 else price
                    volume = float(fields[8]) if len(fields) > 8 else 0.0
                    turnover = float(fields[9]) if len(fields) > 9 else 0.0
                    
                    change_pct = round((price - pre_close) / pre_close * 100, 2) if pre_close > 0 else 0.0

                    data_list.append({
                        '代码': symbol_code,
                        '名称': name,
                        '最新价': price,
                        '涨跌幅': change_pct,
                        '昨收': pre_close,
                        'high': high,
                        'low': low,
                        'volume': volume,
                        'turnover': turnover,
                        '更新时间': '实时'
                    })
                except Exception:
                    continue
            
            return pd.DataFrame(data_list) if data_list else None
        except Exception:
            return None

    @staticmethod
    def _get_tencent_spot_data(symbols: List[str]) -> Optional[pd.DataFrame]:
        try:
            if not symbols:
                return None
            
            req_symbols = DataFetcher._format_symbols(symbols)
            url = f"http://qt.gtimg.cn/q={','.join(req_symbols)}"
            resp = requests.get(url, timeout=5)
            
            if resp.status_code != 200:
                return None

            data_list = []
            for line in resp.text.strip().split('\n'):
                if not line or '=' not in line or '"' not in line:
                    continue
                try:
                    data = line.split('=')[1].strip('\"')
                    parts = data.split('~')
                    if len(parts) < 6:
                        continue

                    name = parts[1].replace('ETF', '').strip()
                    code = parts[2]
                    price = float(parts[3])
                    pre_close = float(parts[4])
                    
                    volume = float(parts[6]) * 100 if len(parts) > 6 and parts[6] else 0.0
                    turnover = float(parts[37]) * 10000 if len(parts) > 37 and parts[37] else 0.0
                    high = float(parts[33]) if len(parts) > 33 and parts[33] else price
                    low = float(parts[34]) if len(parts) > 34 and parts[34] else price
                    
                    change_pct = round((price - pre_close) / pre_close * 100, 2) if pre_close > 0 else 0.0

                    data_list.append({
                        '代码': code,
                        '名称': name,
                        '最新价': price,
                        '涨跌幅': change_pct,
                        '昨收': pre_close,
                        'high': high,
                        'low': low,
                        'volume': volume,
                        'turnover': turnover,
                        '更新时间': '实时'
                    })
                except Exception:
                    continue
                    
            return pd.DataFrame(data_list) if data_list else None
        except Exception:
            return None

    @staticmethod
    def get_etf_hist(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取 ETF 历史行情数据，带备用接口降级
        返回的 df 需要包含列: trade_date, close, open, high, low, turnover, volume, change_pct
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if df is not None and not df.empty:
                    df.rename(columns={
                        '日期': 'trade_date',
                        '收盘': 'close',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '成交额': 'turnover',
                        '成交量': 'volume',
                        '涨跌幅': 'change_pct'
                    }, inplace=True)
                    return df
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"⚠️ Akshare 获取历史行情失败（{symbol}），尝试腾讯财经备用接口...")
                    # 备用方案: 腾讯 K线 API
                    tencent_hist = DataFetcher._get_tencent_hist_data(symbol, start_date, end_date)
                    if tencent_hist is not None and not tencent_hist.empty:
                        print(f"✅ 成功使用腾讯 K 线 API 获取 {symbol} 历史数据")
                        return tencent_hist
        return pd.DataFrame()

    @staticmethod
    def _get_tencent_hist_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            req_symbol = DataFetcher._format_symbols([symbol])[0]
            start_dash = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if len(start_date) == 8 else str(start_date)
            end_dash = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}" if len(end_date) == 8 else str(end_date)

            max_points = 800
            max_pages = 60

            all_items = []
            cur_end = end_dash
            last_earliest = None
            for _ in range(max_pages):
                url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={req_symbol},day,{start_dash},{cur_end},{max_points},qfq"
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    break

                data = resp.json()
                if data.get("code") != 0 or req_symbol not in data.get("data", {}):
                    break

                kline_list = data["data"][req_symbol].get("qfqday", [])
                if not kline_list:
                    kline_list = data["data"][req_symbol].get("day", [])
                if not kline_list:
                    break

                all_items.extend(kline_list)
                earliest = kline_list[0][0] if kline_list and len(kline_list[0]) > 0 else None
                if not earliest:
                    break
                if last_earliest == earliest:
                    break
                last_earliest = earliest

                earliest_ymd = str(earliest).replace("-", "")
                if earliest_ymd <= start_date:
                    break

                import datetime as _dt

                y, m, d = int(earliest_ymd[0:4]), int(earliest_ymd[4:6]), int(earliest_ymd[6:8])
                prev_day = _dt.date(y, m, d) - _dt.timedelta(days=1)
                cur_end = prev_day.isoformat()
                if cur_end < start_dash:
                    break

            if not all_items:
                return None

            records = []
            for item in all_items:
                if len(item) < 6:
                    continue
                date_str = str(item[0])
                ymd = date_str.replace("-", "")
                if ymd < start_date or ymd > end_date:
                    continue
                records.append(
                    {
                        "trade_date": date_str,
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "volume": float(item[5]) * 100,
                        "turnover": 0.0,
                        "change_pct": 0.0,
                    }
                )

            if not records:
                return None

            df = pd.DataFrame(records)
            df.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
            df.sort_values("trade_date", inplace=True)
            df["pre_close"] = df["close"].shift(1)
            df["change_pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
            df.drop(columns=["pre_close"], inplace=True)
            df.fillna({"change_pct": 0.0}, inplace=True)

            return df
        except Exception:
            return None
