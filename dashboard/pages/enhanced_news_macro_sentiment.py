# Enhanced News Fetching and Prompt

def get_comprehensive_asset_news(asset_keywords, extra_sources=None):
    """Enhanced news fetching with multiple sources and better coverage"""
    news = []

    # Expand keywords for better coverage
    expanded_keywords = {
        'GBP': ['GBP', 'Sterling', 'Pound', 'BOE', 'Bank of England', 'UK', 'Britain', 'British', 'Sunak', 'Bailey', 'FTSE', 'Gilt', 'Brexit'],
        'EUR': ['EUR', 'Euro', 'ECB', 'Lagarde', 'European', 'Germany', 'France', 'DAX', 'Bund', 'EU'],
        'USD': ['USD', 'Dollar', 'Fed', 'Powell', 'FOMC', 'Treasury', 'DXY', 'US'],
        'JPY': ['JPY', 'Yen', 'BOJ', 'Ueda', 'Nikkei', 'Japan', 'JGB'],
        'Gold': ['Gold', 'XAU', 'Precious', 'Haven', 'Commodity'],
        'Oil': ['Oil', 'WTI', 'Crude', 'OPEC', 'Energy', 'Brent']
    }

    # Use expanded keywords if available
    for key, expanded in expanded_keywords.items():
        if any(k.upper() in key.upper() for k in asset_keywords):
            asset_keywords.extend(expanded)

    # Remove duplicates
    asset_keywords = list(set(asset_keywords))

    # 1. Finnhub with broader search
    finnhub_df = edm.get_finnhub_headlines(symbols=tuple(asset_keywords[:4]), max_articles=8, category="general")
    if not finnhub_df.empty:
        for _, row in finnhub_df.iterrows():
            news.append(f"- **{row['datetime']}**: [{row['headline']}]({row['url']}) via {row['source']}")

    # 2. NewsAPI with multiple countries
    for country in ['gb', 'us', 'de']:
        articles = edm.get_newsapi_headlines(country=country, category="business", page_size=5)
        for article in articles:
            if any(k.lower() in (article.get('title', '') + article.get('description', '')).lower() for k in asset_keywords):
                news.append(f"- **{article['publishedAt'][:16]}**: [{article['title']}]({article['url']}) via {article['source']['name']}")

    # 3. Try Polygon news if available
    if extra_sources:
        for source in extra_sources:
            poly_news = edm.get_polygon_news(ticker=source, limit=5)
            for art in poly_news:
                news.append(f"- **{art.get('published_utc', '')[:16]}**: [{art['title']}]({art['article_url']}) via {art['publisher']['name']}")

    # Sort by date (newest first) and deduplicate
    news = list(dict.fromkeys(news))  # Remove duplicates while preserving order

    return "\n".join(news[:10]) if news else "⚠️ No recent headlines found. Check news sources."

def fetch_openai_macro_sentiment_v3(snapshot, asset_news, today_econ_events, market_movers, refresh=False):
    """Enhanced prompt with explicit news requirements"""

    # Add more assets to snapshot if not present
    if 'eurgbp' not in snapshot:
        snapshot['eurgbp'] = edm.get_index_quote("EURGBP=X", "EUR/GBP")
    if 'uk10y' not in snapshot:
        uk_yields = edm.get_bond_yields()
        snapshot['uk10y'] = uk_yields.get('GB 10Y', {'current': 'N/A', 'change': 'N/A'})

    # Get comprehensive news for each major currency/asset
    if not asset_news.get('comprehensive_gbp'):
        asset_news['comprehensive_gbp'] = get_comprehensive_asset_news(['GBP'], extra_sources=['FOREX:GBPUSD'])
    if not asset_news.get('comprehensive_eur'):
        asset_news['comprehensive_eur'] = get_comprehensive_asset_news(['EUR'], extra_sources=['FOREX:EURUSD'])

    prompt = f"""You are a professional financial analyst with access to Bloomberg terminal and Reuters. Provide COMPREHENSIVE market analysis.

**CRITICAL**: You MUST include ALL major news impacting each currency/asset. Missing important news is unacceptable.

## 📊 CURRENT MARKET SNAPSHOT
- DXY: {snapshot.get('dxy', {}).get('current', 'N/A')} ({snapshot.get('dxy', {}).get('change', 'N/A'):+})
- VIX: {snapshot.get('vix', {}).get('current', 'N/A')} ({snapshot.get('vix', {}).get('change', 'N/A'):+})
- Gold: ${snapshot.get('gold', {}).get('current', 'N/A')} ({snapshot.get('gold', {}).get('change', 'N/A'):+})
- Oil: ${snapshot.get('oil', {}).get('current', 'N/A')} ({snapshot.get('oil', {}).get('change', 'N/A'):+})
- US 10Y: {snapshot.get('us10y', {}).get('current', 'N/A')}% ({snapshot.get('us10y', {}).get('change', 'N/A'):+})
- EUR/USD: {snapshot.get('eurusd', {}).get('current', 'N/A')} ({snapshot.get('eurusd', {}).get('change', 'N/A'):+})
- GBP/USD: {snapshot.get('gbpusd', {}).get('current', 'N/A')} ({snapshot.get('gbpusd', {}).get('change', 'N/A'):+})
- EUR/GBP: {snapshot.get('eurgbp', {}).get('current', 'N/A')} ({snapshot.get('eurgbp', {}).get('change', 'N/A'):+})
- UK 10Y: {snapshot.get('uk10y', {}).get('current', 'N/A')}% ({snapshot.get('uk10y', {}).get('change', 'N/A'):+})

## 📰 BREAKING NEWS & CATALYSTS

### 💷 GBP - COMPREHENSIVE NEWS
**ALL GBP-related headlines from last 24h:**
{asset_news.get('comprehensive_gbp', 'No GBP news found')}

**Additional GBP catalysts to consider:**
- BOE policy stance and any MPC member comments
- UK economic data (inflation, employment, GDP)
- Political developments (PM statements, fiscal policy)
- Brexit-related news
- FTSE performance impact

### 💶 EUR - COMPREHENSIVE NEWS  
**ALL EUR-related headlines from last 24h:**
{asset_news.get('comprehensive_eur', 'No EUR news found')}

**Additional EUR catalysts:**
- ECB policy and Governing Council comments
- German/French economic data
- EU political developments
- Banking sector news

### Other Key Assets News:
- **DXY/USD**: {asset_news.get('dxy', 'Check Fed speakers, US data')}
- **Gold**: {asset_news.get('gold', 'Check haven flows, real yields')}
- **Oil**: {asset_news.get('oil', 'Check OPEC, inventory data')}
- **VIX**: {asset_news.get('vix', 'Check options flow, event risk')}

## 📈 REQUIRED ANALYSIS FORMAT

### 1. CURRENCY PAIRS - DETAILED
For EACH of: **EUR/USD, GBP/USD, USD/JPY, EUR/GBP**
- **Current Level**: [exact price] ([change] [%change])
- **Intraday Range**: [low] - [high]
- **Key Support**: [level] | **Key Resistance**: [level]
- **Trend**: [Bullish/Bearish/Ranging] because [SPECIFIC catalyst/news]
- **Breaking News Impact**: [How TODAY'S news is affecting price]
- **Trade Setup**: Entry [level], Stop [level], Target [level], R:R [ratio]

### 2. BOND YIELDS & SPREADS
- **US 10Y**: {snapshot.get('us10y', {}).get('current', 'N/A')}% - [interpretation]
- **UK 10Y Gilt**: {snapshot.get('uk10y', {}).get('current', 'N/A')}% - [interpretation]
- **DE 10Y Bund**: {snapshot.get('de10y', {}).get('current', 'N/A')}% - [interpretation]
- **US-DE Spread**: [calculate] bps - [widening/narrowing impact]
- **UK-DE Spread**: [calculate] bps - [Brexit premium analysis]

### 3. RISK SENTIMENT INDICATORS
- **VIX Level**: {snapshot.get('vix', {}).get('current', 'N/A')} - [fear/greed interpretation]
- **Gold/Oil Ratio**: [calculate] - [risk on/off signal]
- **JPY Performance**: [haven demand analysis]
- **Equity Futures**: [risk appetite gauge]

### 4. CENTRAL BANK WATCH
**Next 72 hours:**
- Fed: [speakers/data]
- ECB: [speakers/data]
- BOE: [speakers/data]
- BOJ: [speakers/data]

### 5. TODAY'S KEY ECONOMIC RELEASES
{today_econ_events if not today_econ_events.empty else "No major releases today"}

### 6. CROSS-ASSET CORRELATIONS
- [Which assets are moving together/diverging TODAY]
- [Unusual correlations to watch]

### 7. TOP 3 TRADING OPPORTUNITIES
1. **[Asset]**: [Detailed setup with entry, stop, target, catalyst]
2. **[Asset]**: [Detailed setup with entry, stop, target, catalyst]
3. **[Asset]**: [Detailed setup with entry, stop, target, catalyst]

### 8. RISK WARNINGS
- [Key event risk in next 24-48h]
- [Technical levels that could trigger stops]
- [Correlation breakdown risks]

**Remember**: Be SPECIFIC with numbers, times, and reasons. NO generic statements. If major news is happening (like BOE decisions, GDP releases, political events), it MUST be prominently featured."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a Bloomberg terminal with real-time access to all financial news and data. Provide specific, tradeable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"

# Also update the get_high_impact_news function to be more comprehensive
def get_comprehensive_market_news():
    """Fetch high-impact news from multiple sources"""
    all_news = []

    # Expanded keywords for different news types
    keywords_by_category = {
        'central_banks': ["FOMC", "ECB", "BOE", "BOJ", "rate", "Powell", "Lagarde", "Bailey", "Ueda", "hawkish", "dovish"],
        'economic_data': ["CPI", "NFP", "GDP", "unemployment", "inflation", "PMI", "retail sales", "jobs"],
        'geopolitical': ["war", "sanctions", "trade", "tariff", "election", "crisis"],
        'market_moves': ["crash", "rally", "surge", "plunge", "breakout", "volatility", "record"],
        'corporate': ["earnings", "merger", "bankruptcy", "IPO", "buyback"],
        'currency_specific': ["dollar", "euro", "pound", "yen", "sterling", "currency"]
    }

    try:
        # 1. Finnhub general news
        response = requests.get(
            "https://finnhub.io/api/v1/news?category=general&token=" + st.secrets.get("finnhub_api_key", ""),
            timeout=10
        )
        if response.ok:
            articles = response.json()
            for article in articles[:20]:  # Check more articles
                headline = article.get("headline", "").lower()
                # Check against all keyword categories
                for category, keywords in keywords_by_category.items():
                    if any(k.lower() in headline for k in keywords):
                        all_news.append({
                            'category': category,
                            'datetime': datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M'),
                            'headline': article['headline'],
                            'url': article['url'],
                            'source': article.get('source', 'Unknown')
                        })
                        break

        # 2. NewsAPI for multiple countries
        for country in ['us', 'gb', 'de']:
            response = requests.get(
                f"https://newsapi.org/v2/top-headlines?country={country}&category=business&apiKey=" + st.secrets.get("newsapi_key", ""),
                timeout=10
            )
            if response.ok:
                data = response.json()
                for article in data.get('articles', [])[:10]:
                    all_news.append({
                        'category': 'regional',
                        'datetime': article['publishedAt'][:16],
                        'headline': article['title'],
                        'url': article['url'],
                        'source': article['source']['name']
                    })

        # Sort by datetime and remove duplicates
        all_news = sorted(all_news, key=lambda x: x['datetime'], reverse=True)

        # Remove duplicates based on headline similarity
        unique_news = []
        seen_headlines = set()
        for item in all_news:
            headline_words = set(item['headline'].lower().split())
            if not any(len(headline_words & seen_words) > 3 for seen_words in seen_headlines):
                unique_news.append(item)
                seen_headlines.add(frozenset(headline_words))

        return unique_news[:15]  # Return top 15 unique news items

    except Exception as e:
        st.error(f"Error fetching comprehensive news: {e}")
        return []