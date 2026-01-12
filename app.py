import os
import json
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# Config - Use local files for standalone deployment
METADATA_FILE = os.path.join(os.path.dirname(__file__), 'master_metadata.json')
SYMBOL_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'symbol_cache.json')

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def resolve_ticker(t, symbol_cache):
    if t in symbol_cache:
        return symbol_cache[t]
    if "-" in t:
        parts = t.split("-")
        return f"{parts[0]}-P{parts[1]}"
    return t

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tickers')
def get_tickers():
    metadata = load_json(METADATA_FILE)
    # Return basic info for search/select
    tickers = []
    for t, m in metadata.items():
        tickers.append({
            "ticker": t,
            "name": m.get('name', ''),
            "sector": m.get('sector', 'Other')
        })
    return jsonify(tickers)

@app.route('/api/sectors')
def get_sectors():
    metadata = load_json(METADATA_FILE)
    sectors = sorted(list(set(m.get('sector', 'Other') for m in metadata.values() if m.get('sector'))))
    return jsonify(sectors)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    target_ticker = data.get('target')
    peer_tickers = data.get('peers', [])
    ma_period = int(data.get('ma_period', 20))
    lookback = data.get('lookback', '2y')
    
    if not target_ticker or not peer_tickers:
        return jsonify({"error": "Missing target or peers"}), 400

    symbol_cache = load_json(SYMBOL_CACHE_FILE)
    target_y = resolve_ticker(target_ticker, symbol_cache)
    peers_y = [resolve_ticker(p, symbol_cache) for p in peer_tickers]
    all_symbols = list(set([target_y] + peers_y))

    try:
        # Fetch historical data
        df = yf.download(all_symbols, period=lookback, progress=False, threads=True, group_by='ticker')
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        # Extract Close prices
        closes = pd.DataFrame()
        if len(all_symbols) == 1:
            closes[target_ticker] = df['Close']
        else:
            for t_raw in all_symbols:
                # Map back to user ticker if possible
                user_t = next((k for k,v in symbol_cache.items() if v == t_raw), t_raw)
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if t_raw in df.columns.levels[0]:
                            closes[user_t] = df[t_raw]['Close']
                    elif t_raw in df.columns:
                        closes[user_t] = df[t_raw]
                except:
                    continue

        if target_ticker not in closes.columns:
            return jsonify({"error": "Target data could not be retrieved"}), 404

        # Calculate SMAs for smoothing individually
        # This prevents a single ticker with missing data from breaking the whole process
        smas_dict = {}
        for col in closes.columns:
            s_ma = closes[col].rolling(window=ma_period).mean()
            if not s_ma.dropna().empty:
                smas_dict[col] = s_ma
        
        if target_ticker not in smas_dict:
            return jsonify({"error": f"Target {target_ticker} does not have enough data for a {ma_period}-day MA"}), 400

        target_sma = smas_dict[target_ticker]
        results = []

        for peer, peer_sma_full in smas_dict.items():
            if peer == target_ticker:
                continue
            
            # Align target and peer
            combined = pd.DataFrame({'t': target_sma, 'p': peer_sma_full}).dropna()
            
            if len(combined) < 10: # Minimum overlap required
                continue

            t_aligned = combined['t']
            p_aligned = combined['p']
            
            # Correlation Analysis
            corr = t_aligned.corr(p_aligned)
            if pd.isna(corr):
                continue

            # Lag Calculation
            ratio = t_aligned / p_aligned
            hist_mean_ratio = ratio.mean()
            current_ratio = ratio.iloc[-1]
            deviation = (current_ratio / hist_mean_ratio - 1) * 100

            results.append({
                "ticker": peer,
                "correlation": round(float(corr), 4),
                "deviation": round(float(deviation), 2),
                "current_price": round(float(closes[peer].iloc[-1]), 2),
                "target_price": round(float(closes[target_ticker].iloc[-1]), 2)
            })

        # Sort by correlation descending
        results.sort(key=lambda x: x['correlation'], reverse=True)

        # Prepare sanitized values for JSON (convert NaN to None/null)
        def fix_val(v, precision=3):
            if pd.isna(v): return None
            return round(float(v), precision)

        sanitized_results = []
        for r in results:
            sanitized_results.append({
                "ticker": r['ticker'],
                "correlation": fix_val(r['correlation'], 4),
                "deviation": fix_val(r['deviation'], 2),
                "current_price": fix_val(r['current_price'], 2),
                "target_price": fix_val(r['target_price'], 2)
            })

        return jsonify({
            "target": target_ticker,
            "period": lookback,
            "ma": ma_period,
            "results": sanitized_results,
            "target_series": {
                "labels": [d.strftime('%Y-%m-%d') for d in target_sma.index],
                "values": [fix_val(v) for v in target_sma.values]
            },
            # Include peer series for ALL results so any row can be clicked for chart
            "peer_series": {
                r['ticker']: [fix_val(v) for v in smas_dict[r['ticker']].values]
                for r in results
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/best-pairs', methods=['POST'])
def best_pairs():
    data = request.json
    sector = data.get('sector', 'Financial')
    ma_period = int(data.get('ma_period', 20))
    limit = int(data.get('limit', 20))

    metadata = load_json(METADATA_FILE)
    symbol_cache = load_json(SYMBOL_CACHE_FILE)
    
    # Filter tickers by sector
    sector_tickers = [t for t, m in metadata.items() if m.get('sector') == sector]
    if not sector_tickers:
        return jsonify({"error": f"No tickers found for sector {sector}"}), 404
    
    # Limit to top 60 tickers in sector to keep it under ~1800 pairs
    sector_tickers = sector_tickers[:60]
    
    all_symbols = [resolve_ticker(t, symbol_cache) for t in sector_tickers]

    try:
        df = yf.download(all_symbols, period="2y", progress=False, threads=True, group_by='ticker')
        if df.empty:
            return jsonify({"error": "No data found for this sector"}), 404

        closes = pd.DataFrame()
        for t_raw in all_symbols:
            user_t = next((k for k,v in symbol_cache.items() if v == t_raw), t_raw)
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t_raw in df.columns.levels[0]:
                        closes[user_t] = df[t_raw]['Close']
                elif t_raw in df.columns:
                    closes[user_t] = df[t_raw]
            except: continue

        # Individual SMAs
        smas_dict = {}
        for col in closes.columns:
            s_ma = closes[col].rolling(window=ma_period).mean()
            if not s_ma.dropna().empty:
                smas_dict[col] = s_ma

        # All-vs-All Correlation
        processed_df = pd.DataFrame(smas_dict).dropna(how='all')
        corr_matrix = processed_df.corr()
        
        pairs = []
        seen_pairs = set()
        cols = corr_matrix.columns
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                t1, t2 = cols[i], cols[j]
                correlation = corr_matrix.iloc[i, j]
                
                if pd.isna(correlation) or correlation < 0.7:
                    continue
                
                # Calculate current lag for this pair
                # We'll use t1 as target and t2 as peer
                s1 = smas_dict[t1]
                s2 = smas_dict[t2]
                combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
                if len(combined) < 10: continue
                
                ratio = combined['s1'] / combined['s2']
                curr_div = (ratio.iloc[-1] / ratio.mean() - 1) * 100
                
                pairs.append({
                    "ticker1": t1,
                    "ticker2": t2,
                    "correlation": round(float(correlation), 4),
                    "deviation": round(float(curr_div), 2)
                })

        # Sort by correlation
        pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        return jsonify({
            "sector": sector,
            "count": len(pairs),
            "results": pairs[:limit]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Railway provides the PORT environment variable
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
