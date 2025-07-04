
//+------------------------------------------------------------------+
//|                                             TickDataFileWriter.mq5 |
//|                        Copyright 2025, Abacus.AI Data Services |
//|                                         https://www.abacus.ai |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Abacus.AI Data Services"
#property link      "https://www.abacus.ai"
#property version   "1.00"

//--- input parameters
input string InpFileName = "tick_data.csv"; // The name of the CSV file to write to

int fileHandle; // File handle

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Open the file to write data
   fileHandle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');

//--- If the file failed to open, print an error and stop
   if(fileHandle == INVALID_HANDLE)
     {
      Print("Error opening file ", InpFileName, ". Error code: ", GetLastError());
      return(INIT_FAILED);
     }

//--- Write the header row to the CSV file
   FileWriteString(fileHandle, "timestamp,bid,ask,last,volume,symbol\n");
   Print("TickDataFileWriter started. Writing data to ", InpFileName);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Close the file when the EA is removed or the terminal closes
   if(fileHandle != INVALID_HANDLE)
     {
      FileClose(fileHandle);
      Print("TickDataFileWriter stopped. File closed.");
     }
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Get the latest tick information
   MqlTick last_tick;
   if(!SymbolInfoTick(_Symbol, last_tick))
     {
      Print("SymbolInfoTick() failed, error code: ", GetLastError());
      return;
     }

//--- Prepare the data string
   string data_row = TimeToString(last_tick.time_msc, TIME_DATE|TIME_SECONDS|TIME_MILLISECONDS) + "," +
                     DoubleToString(last_tick.bid, _Digits) + "," +
                     DoubleToString(last_tick.ask, _Digits) + "," +
                     DoubleToString(last_tick.last, _Digits) + "," +
                     IntegerToString(last_tick.volume) + "," +
                     _Symbol;

//--- Write the data to the file and force it to disk
   if(fileHandle != INVALID_HANDLE)
     {
      FileWriteString(fileHandle, data_row + "\n");
      FileFlush(fileHandle); // Ensure data is written immediately
     }
  }
//+------------------------------------------------------------------+
