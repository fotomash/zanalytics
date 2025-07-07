//+------------------------------------------------------------------+
//|                                    MultiSymbolSender.mq5         |
//|                              ZANFLOW Multi-Symbol Trading System |
//+------------------------------------------------------------------+
#property copyright "ZANFLOW Trading System"
#property version   "2.00"

// Input parameters
input string   InpURL = "http://127.0.0.1:8080/webhook";  // Webhook URL
input bool     InpAllSymbols = true;                      // Send all symbols from Market Watch
input string   InpSymbols = "EURUSD,GBPUSD,USDJPY";      // Specific symbols (if not all)
input int      InpTimer = 1;                               // Timer interval in seconds
input bool     InpDebug = false;                          // Enable debug output

// Global variables
string g_symbols[];
int g_symbolCount;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Get symbols to monitor
    if(InpAllSymbols)
    {
        GetMarketWatchSymbols();
    }
    else
    {
        ParseSymbolList(InpSymbols);
    }

    Print("MultiSymbolSender initialized with ", g_symbolCount, " symbols");

    // Set timer
    EventSetTimer(InpTimer);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Get all symbols from Market Watch                               |
//+------------------------------------------------------------------+
void GetMarketWatchSymbols()
{
    g_symbolCount = 0;
    ArrayResize(g_symbols, 0);

    int total = SymbolsTotal(true);  // true = only symbols in Market Watch

    for(int i = 0; i < total; i++)
    {
        string symbol = SymbolName(i, true);
        if(symbol != "")
        {
            ArrayResize(g_symbols, g_symbolCount + 1);
            g_symbols[g_symbolCount] = symbol;
            g_symbolCount++;
        }
    }

    Print("Found ", g_symbolCount, " symbols in Market Watch");
}

//+------------------------------------------------------------------+
//| Parse comma-separated symbol list                                |
//+------------------------------------------------------------------+
void ParseSymbolList(string symbolList)
{
    g_symbolCount = 0;
    ArrayResize(g_symbols, 0);

    string sep = ",";
    ushort u_sep = StringGetCharacter(sep, 0);
    string result[];

    int k = StringSplit(symbolList, u_sep, result);

    for(int i = 0; i < k; i++)
    {
        string symbol = result[i];
        StringTrimLeft(symbol);
        StringTrimRight(symbol);

        if(symbol != "" && SymbolSelect(symbol, true))
        {
            ArrayResize(g_symbols, g_symbolCount + 1);
            g_symbols[g_symbolCount] = symbol;
            g_symbolCount++;
        }
    }
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    SendAllSymbolsData();
}

//+------------------------------------------------------------------+
//| Calculate volume metrics                                         |
//+------------------------------------------------------------------+
double CalculateVolume(string symbol, int period_seconds)
{
    MqlTick ticks[];
    datetime from = TimeCurrent() - period_seconds;
    datetime to = TimeCurrent();

    int copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, from * 1000, to * 1000);

    if(copied > 0)
    {
        double totalVolume = 0;
        for(int i = 0; i < copied; i++)
        {
            totalVolume += ticks[i].volume;
        }

        // Return volume per minute
        return totalVolume / (period_seconds / 60.0);
    }

    return 0;
}

//+------------------------------------------------------------------+
//| Send data for all symbols                                       |
//+------------------------------------------------------------------+
void SendAllSymbolsData()
{
    // Build JSON array
    string json = "{\"symbols\": [";

    for(int i = 0; i < g_symbolCount; i++)
    {
        string symbol = g_symbols[i];

        // Get tick data
        MqlTick tick;
        if(!SymbolInfoTick(symbol, tick))
            continue;

        // Get additional data
        int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

        // Calculate volumes
        double tickVolume = tick.volume;
        double volumePerMin = CalculateVolume(symbol, 60);  // Last minute
        double volume5Min = CalculateVolume(symbol, 300);   // Last 5 minutes

        // Get recent candles
        MqlRates rates[];
        ArraySetAsSeries(rates, true);
        int copied = CopyRates(symbol, PERIOD_M1, 0, 5, rates);

        if(i > 0) json += ",";

        json += "{";
        json += "\"symbol\": \"" + symbol + "\",";
        json += "\"timestamp\": " + IntegerToString(tick.time) + ",";
        json += "\"bid\": " + DoubleToString(tick.bid, digits) + ",";
        json += "\"ask\": " + DoubleToString(tick.ask, digits) + ",";
        json += "\"last\": " + DoubleToString(tick.last, digits) + ",";
        json += "\"spread\": " + IntegerToString(spread) + ",";
        json += "\"volume\": {";
        json += "  \"tick\": " + DoubleToString(tickVolume, 0) + ",";
        json += "  \"per_minute\": " + DoubleToString(volumePerMin, 2) + ",";
        json += "  \"last_5min\": " + DoubleToString(volume5Min, 2);
        json += "},";

        // Add candle data
        if(copied > 0)
        {
            json += "\"candles\": [";
            for(int j = 0; j < copied && j < 5; j++)
            {
                if(j > 0) json += ",";
                json += "{";
                json += "\"time\": " + IntegerToString(rates[j].time) + ",";
                json += "\"open\": " + DoubleToString(rates[j].open, digits) + ",";
                json += "\"high\": " + DoubleToString(rates[j].high, digits) + ",";
                json += "\"low\": " + DoubleToString(rates[j].low, digits) + ",";
                json += "\"close\": " + DoubleToString(rates[j].close, digits) + ",";
                json += "\"tick_volume\": " + IntegerToString(rates[j].tick_volume) + ",";
                json += "\"real_volume\": " + DoubleToString(rates[j].real_volume, 0);
                json += "}";
            }
            json += "]";
        }

        json += "}";
    }

    json += "], \"account\": {";
    json += "\"balance\": " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
    json += "\"equity\": " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
    json += "\"margin\": " + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN), 2) + ",";
    json += "\"free_margin\": " + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2);
    json += "}}";

    // Send HTTP request
    SendHttpRequest(json);
}

//+------------------------------------------------------------------+
//| Send HTTP POST request                                           |
//+------------------------------------------------------------------+
void SendHttpRequest(string jsonData)
{
    char post[];
    char result[];
    string headers = "Content-Type: application/json\r\n";

    StringToCharArray(jsonData, post, 0, StringLen(jsonData));

    int res = WebRequest(
        "POST",
        InpURL,
        headers,
        5000,
        post,
        result,
        headers
    );

    if(res == -1)
    {
        int error = GetLastError();
        Print("WebRequest failed with error: ", error);
    }
    else
    {
        if(InpDebug)
            Print("✅ Sent data for ", g_symbolCount, " symbols");
    }
}
//+------------------------------------------------------------------+
