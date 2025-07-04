//+------------------------------------------------------------------+
//|                                               HttpsJsonSender.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

// Input parameters
input string   API_URL = "https://your-ngrok-url.ngrok.io/api/market_data";  // API URL to send data to
input int      UPDATE_INTERVAL = 60;                                          // Update interval in seconds
input bool     SEND_TECHNICAL_INDICATORS = true;                              // Send technical indicators
input bool     SEND_PRICE_DATA = true;                                        // Send price data
input bool     SEND_VOLUME_DATA = true;                                       // Send volume data

// Global variables
int lastUpdateTime = 0;
string jsonData = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Enable WebRequest for the specified URL
   string terminal_data_path = TerminalInfoString(TERMINAL_DATA_PATH);
   string filename = terminal_data_path + "\\MQL5\\Files\\WebRequestAllowed.txt";

   // Add URL to the list of allowed URLs
   if(!FileIsExist(filename))
   {
      int file = FileOpen(filename, FILE_WRITE|FILE_TXT);
      if(file != INVALID_HANDLE)
      {
         FileWriteString(file, API_URL + "\n");
         FileClose(file);
         Print("Added API URL to WebRequestAllowed.txt");
      }
      else
      {
         Print("Failed to open WebRequestAllowed.txt, error: ", GetLastError());
         return(INIT_FAILED);
      }
   }

   Print("HttpsJsonSender initialized successfully");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("HttpsJsonSender deinitialized, reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only update data at the specified interval
   if(TimeCurrent() - lastUpdateTime < UPDATE_INTERVAL)
      return;

   lastUpdateTime = (int)TimeCurrent();

   // Prepare and send JSON data
   if(PrepareJsonData())
   {
      SendJsonData();
   }
}

//+------------------------------------------------------------------+
//| Prepare JSON data with current market information                |
//+------------------------------------------------------------------+
bool PrepareJsonData()
{
   // Reset JSON string
   jsonData = "";

   // Start JSON object
   jsonData = "{";

   // Add basic market information
   jsonData += "\"symbol\":\"" + Symbol() + "\",";
   jsonData += "\"timestamp\":" + IntegerToString(TimeCurrent()) + ",";
   jsonData += "\"server_time\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";

   if(SEND_PRICE_DATA)
   {
      // Get latest price data
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(Symbol(), PERIOD_CURRENT, 0, 10, rates);

      if(copied > 0)
      {
         jsonData += "\"price_data\":{";
         jsonData += "\"open\":" + DoubleToString(rates[0].open, 8) + ",";
         jsonData += "\"high\":" + DoubleToString(rates[0].high, 8) + ",";
         jsonData += "\"low\":" + DoubleToString(rates[0].low, 8) + ",";
         jsonData += "\"close\":" + DoubleToString(rates[0].close, 8) + ",";
         jsonData += "\"spread\":" + IntegerToString(SymbolInfoInteger(Symbol(), SYMBOL_SPREAD)) + ",";
         jsonData += "\"point\":" + DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_POINT), 8) + ",";
         jsonData += "\"digits\":" + IntegerToString(SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));
         jsonData += "},";
      }
      else
      {
         Print("Failed to copy rates data, error: ", GetLastError());
         return false;
      }
   }

   if(SEND_VOLUME_DATA)
   {
      // Get volume data
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(Symbol(), PERIOD_CURRENT, 0, 10, rates);

      if(copied > 0)
      {
         jsonData += "\"volume_data\":{";
         jsonData += "\"tick_volume\":" + IntegerToString(rates[0].tick_volume) + ",";
         jsonData += "\"real_volume\":" + IntegerToString(rates[0].real_volume);
         jsonData += "},";
      }
      else
      {
         Print("Failed to copy volume data, error: ", GetLastError());
         return false;
      }
   }

   if(SEND_TECHNICAL_INDICATORS)
   {
      // Calculate technical indicators
      jsonData += "\"indicators\":{";

      // Moving Averages
      double ma_fast[], ma_slow[];
      ArraySetAsSeries(ma_fast, true);
      ArraySetAsSeries(ma_slow, true);

      int ma_fast_handle = iMA(Symbol(), PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
      int ma_slow_handle = iMA(Symbol(), PERIOD_CURRENT, 50, 0, MODE_SMA, PRICE_CLOSE);

      if(ma_fast_handle != INVALID_HANDLE && ma_slow_handle != INVALID_HANDLE)
      {
         bool ma_fast_copied = CopyBuffer(ma_fast_handle, 0, 0, 3, ma_fast) > 0;
         bool ma_slow_copied = CopyBuffer(ma_slow_handle, 0, 0, 3, ma_slow) > 0;

         if(ma_fast_copied && ma_slow_copied)
         {
            jsonData += "\"ma_fast\":" + DoubleToString(ma_fast[0], 8) + ",";
            jsonData += "\"ma_slow\":" + DoubleToString(ma_slow[0], 8) + ",";
         }
      }

      // RSI
      double rsi[];
      ArraySetAsSeries(rsi, true);
      int rsi_handle = iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE);

      if(rsi_handle != INVALID_HANDLE)
      {
         if(CopyBuffer(rsi_handle, 0, 0, 3, rsi) > 0)
         {
            jsonData += "\"rsi\":" + DoubleToString(rsi[0], 8) + ",";
         }
      }

      // Bollinger Bands
      double bb_upper[], bb_lower[], bb_middle[];
      ArraySetAsSeries(bb_upper, true);
      ArraySetAsSeries(bb_lower, true);
      ArraySetAsSeries(bb_middle, true);

      int bb_handle = iBands(Symbol(), PERIOD_CURRENT, 20, 2, 0, PRICE_CLOSE);

      if(bb_handle != INVALID_HANDLE)
      {
         bool bb_middle_copied = CopyBuffer(bb_handle, 0, 0, 3, bb_middle) > 0;
         bool bb_upper_copied = CopyBuffer(bb_handle, 1, 0, 3, bb_upper) > 0;
         bool bb_lower_copied = CopyBuffer(bb_handle, 2, 0, 3, bb_lower) > 0;

         if(bb_middle_copied && bb_upper_copied && bb_lower_copied)
         {
            jsonData += "\"bb_upper\":" + DoubleToString(bb_upper[0], 8) + ",";
            jsonData += "\"bb_middle\":" + DoubleToString(bb_middle[0], 8) + ",";
            jsonData += "\"bb_lower\":" + DoubleToString(bb_lower[0], 8) + ",";
         }
      }

      // MACD
      double macd_main[], macd_signal[];
      ArraySetAsSeries(macd_main, true);
      ArraySetAsSeries(macd_signal, true);

      int macd_handle = iMACD(Symbol(), PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);

      if(macd_handle != INVALID_HANDLE)
      {
         bool macd_main_copied = CopyBuffer(macd_handle, 0, 0, 3, macd_main) > 0;
         bool macd_signal_copied = CopyBuffer(macd_handle, 1, 0, 3, macd_signal) > 0;

         if(macd_main_copied && macd_signal_copied)
         {
            jsonData += "\"macd_main\":" + DoubleToString(macd_main[0], 8) + ",";
            jsonData += "\"macd_signal\":" + DoubleToString(macd_signal[0], 8) + ",";
            jsonData += "\"macd_histogram\":" + DoubleToString(macd_main[0] - macd_signal[0], 8);
         }
      }

      // Remove trailing comma if present
      if(StringSubstr(jsonData, StringLen(jsonData) - 1, 1) == ",")
         jsonData = StringSubstr(jsonData, 0, StringLen(jsonData) - 1);

      jsonData += "}";
   }

   // Remove trailing comma if present
   if(StringSubstr(jsonData, StringLen(jsonData) - 1, 1) == ",")
      jsonData = StringSubstr(jsonData, 0, StringLen(jsonData) - 1);

   // Close JSON object
   jsonData += "}";

   return true;
}

//+------------------------------------------------------------------+
//| Send JSON data to the API server                                 |
//+------------------------------------------------------------------+
void SendJsonData()
{
   char data[];
   StringToCharArray(jsonData, data);

   char result[];
   string result_headers;

   // Define headers
   string headers = "Content-Type: application/json\r\n";

   int res = WebRequest("POST", API_URL, headers, 10000, data, result, result_headers);

   if(res == -1)
   {
      int error_code = GetLastError();
      Print("WebRequest failed with error code: ", error_code);

      if(error_code == 4060)
      {
         Print("Make sure URL '", API_URL, "' is added to the list of allowed URLs in Tools > Options > Expert Advisors");
      }
   }
   else
   {
      string result_string = CharArrayToString(result);
      Print("Data sent successfully, server response: ", result_string);
   }
}
