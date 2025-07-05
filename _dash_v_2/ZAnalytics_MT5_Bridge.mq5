//+------------------------------------------------------------------+
//| ZAnalytics_MT5_Bridge.mq5                                        |
//| Simplified version that avoids problematic string functions      |
//+------------------------------------------------------------------+
#property copyright "ZAnalytics"
#property version   "4.00"
#property strict

#include <Trade\Trade.mqh>

// Input parameters
input string API_URL = "http://localhost:8000/api/v1/ingest/candle";
input string SYMBOLS = "XAUUSD";  // Comma-separated symbols
input ENUM_TIMEFRAMES TIMEFRAME = PERIOD_H1;
input bool ENABLE_TICK_DATA = true;
input bool ENABLE_CANDLE_DATA = true;
input string EXPORT_PATH = "C:\\MT5_Data\\Exports\\";
input int MIN_MILLISECONDS_BETWEEN_SENDS = 100;
input bool ENABLE_DETAILED_LOGGING = true;

// Global variables
CTrade trade;
datetime lastBarTime[];
string symbolList[];
int symbolCount = 0;
datetime lastSendTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Parse symbols
    ParseSymbols();
    
    // Initialize last bar times
    ArrayResize(lastBarTime, symbolCount);
    for(int i = 0; i < symbolCount; i++)
    {
        lastBarTime[i] = 0;
    }
    
    // Create export directory
    string folder = EXPORT_PATH;
    if(!FolderCreate(folder))
    {
        Print("Failed to create folder: ", folder, ", Error: ", GetLastError());
    }
    
    Print("ZAnalytics Bridge initialized for ", symbolCount, " symbols");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(ENABLE_TICK_DATA)
    {
        ProcessTickData();
    }
    
    if(ENABLE_CANDLE_DATA)
    {
        ProcessCandleData();
    }
}

//+------------------------------------------------------------------+
//| Process tick data                                                |
//+------------------------------------------------------------------+
void ProcessTickData()
{
    // Rate limiting
    datetime currentTime = TimeLocal();
    if(currentTime - lastSendTime < MIN_MILLISECONDS_BETWEEN_SENDS/1000)
        return;
        
    for(int i = 0; i < symbolCount; i++)
    {
        string symbol = symbolList[i];
        
        // Get current tick
        MqlTick tick;
        if(SymbolInfoTick(symbol, tick))
        {
            // Export to CSV
            ExportTickToCSV(symbol, tick);
            
            lastSendTime = TimeLocal();
        }
    }
}

//+------------------------------------------------------------------+
//| Process candle data                                              |
//+------------------------------------------------------------------+
void ProcessCandleData()
{
    for(int i = 0; i < symbolCount; i++)
    {
        string symbol = symbolList[i];
        
        // Check if new bar formed
        datetime currentBarTime = iTime(symbol, TIMEFRAME, 0);
        
        if(currentBarTime != lastBarTime[i])
        {
            if(lastBarTime[i] != 0)  // Skip first run
            {
                // Get completed bar data (index 1)
                double open = iOpen(symbol, TIMEFRAME, 1);
                double high = iHigh(symbol, TIMEFRAME, 1);
                double low = iLow(symbol, TIMEFRAME, 1);
                double close = iClose(symbol, TIMEFRAME, 1);
                long volume = iVolume(symbol, TIMEFRAME, 1);
                datetime barTime = iTime(symbol, TIMEFRAME, 1);
                
                // Export to CSV
                ExportCandleToCSV(symbol, barTime, open, high, low, close, volume);
                
                // Log
                if(ENABLE_DETAILED_LOGGING)
                    Print("Processed ", symbol, " bar at ", TimeToString(barTime));
            }
            
            lastBarTime[i] = currentBarTime;
        }
    }
}

//+------------------------------------------------------------------+
//| Export tick to CSV                                               |
//+------------------------------------------------------------------+
void ExportTickToCSV(string symbol, MqlTick &tick)
{
    string filename = EXPORT_PATH + symbol + "_ticks.csv";
    
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON, ",");
    if(handle != INVALID_HANDLE)
    {
        // Write header if new file
        if(FileSize(handle) == 0)
        {
            FileWrite(handle, "timestamp", "symbol", "bid", "ask", "last", "volume");
        }
        
        string timestamp = TimeToString(tick.time, TIME_DATE|TIME_SECONDS);
        
        // Format timestamp - using direct concatenation instead of StringConcatenate
        string formattedTime = StringSubstr(timestamp, 0, 4) + "-" + 
                              StringSubstr(timestamp, 5, 2) + "-" + 
                              StringSubstr(timestamp, 8, 2) + "T" + 
                              StringSubstr(timestamp, 11) + ".000000";
        
        FileWrite(handle, formattedTime, symbol, 
                 DoubleToString(tick.bid, 5), DoubleToString(tick.ask, 5), 
                 DoubleToString(tick.last, 5), IntegerToString(tick.volume));
        
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Export candle to CSV                                             |
//+------------------------------------------------------------------+
void ExportCandleToCSV(string symbol, datetime barTime, double open, double high, double low, double close, long volume)
{
    string filename = EXPORT_PATH + symbol + "_" + StringSubstr(EnumToString(TIMEFRAME), 7) + "_candles.csv";
    
    int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON, ",");
    if(handle != INVALID_HANDLE)
    {
        // Write header if new file
        if(FileSize(handle) == 0)
        {
            FileWrite(handle, "timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume");
        }
        
        string timestamp = TimeToString(barTime, TIME_DATE|TIME_SECONDS);
        
        // Format timestamp - using direct concatenation instead of StringConcatenate
        string formattedTime = StringSubstr(timestamp, 0, 4) + "-" + 
                              StringSubstr(timestamp, 5, 2) + "-" + 
                              StringSubstr(timestamp, 8, 2) + "T" + 
                              StringSubstr(timestamp, 11) + ".000000";
        
        string tf = StringSubstr(EnumToString(TIMEFRAME), 7);
        
        FileWrite(handle, formattedTime, symbol, tf, 
                 DoubleToString(open, 5), DoubleToString(high, 5), 
                 DoubleToString(low, 5), DoubleToString(close, 5), 
                 IntegerToString(volume));
        
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Parse symbols from input string                                  |
//+------------------------------------------------------------------+
void ParseSymbols()
{
    string temp = SYMBOLS;
    symbolCount = 0;
    
    // Count symbols
    for(int i = 0; i < StringLen(temp); i++)
    {
        if(StringGetCharacter(temp, i) == ',')
            symbolCount++;
    }
    symbolCount++; // Add one for last symbol
    
    // Resize array
    ArrayResize(symbolList, symbolCount);
    
    // Parse symbols
    int pos = 0;
    for(int i = 0; i < symbolCount; i++)
    {
        int nextComma = StringFind(temp, ",", pos);
        if(nextComma == -1)
            nextComma = StringLen(temp);
        
        symbolList[i] = StringSubstr(temp, pos, nextComma - pos);
        pos = nextComma + 1;
    }
}