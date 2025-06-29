// +------------------------------------------------------------------+
// | ncOS_MT5_TickExporter_VolumeEnhanced.mq5 |
// +------------------------------------------------------------------+
# property copyright "ncOS Framework"
# property version   "4.00"

// Input
parameters
input
int
ExportIntervalMinutes = 1; // Auto - export
interval in minutes
input
bool
ExportAllSymbols = true; // Export
all
symbols(true) or current
only(false)
input
bool
OverwriteFile = true; // Overwrite
file(true) or create
timestamped(false)
input
bool
ExportRawFormat = false; // Export
raw
format(true) or standard
CSV(false)
input
bool
EnableAutoExport = true; // Enable
automatic
export
input
int
TicksLookbackMinutes = 60; // How
many
minutes
of
tick
history
to
export
input
int
MaxTicksPerExport = 100000; // Maximum
ticks
per
export
input
string
ExportFolder = "TickData"; // Export
folder
name
input
bool
CalculateVolume = true; // Calculate
inferred
volume
metrics
input
double
VolumeMultiplier = 100.0; // Base
volume
for tick volume calculation

// Global
variables
datetime
lastExportTime = 0;

// +------------------------------------------------------------------+
// | Expert
initialization
function |
// +------------------------------------------------------------------+
int
OnInit()
{
// Create
export
directory
if (!CreateExportDirectory())
{
    Print("Failed to create export directory");
return (INIT_FAILED);
}

// Set
timer
for auto - export
if (EnableAutoExport)
{
EventSetTimer(ExportIntervalMinutes * 60);
}

Print("TickDataExporter with Volume Analysis initialized");
Print("Export Folder: ", ExportFolder);
Print("Calculate Volume: ", CalculateVolume ? "YES": "NO");
Print("Volume Multiplier: ", VolumeMultiplier);

// Do
initial
export
ExportTickData();

return (INIT_SUCCEEDED);
}

// +------------------------------------------------------------------+
   // | Calculate
inferred
volume
from tick data |
// +------------------------------------------------------------------+
double
CalculateInferredVolume(MqlTick & tick, MqlTick & prevTick, double
baseVolume)
{
double
volume = baseVolume;

// 1.
Price
change
factor(larger
moves = more
volume)
double
priceChange = MathAbs(tick.bid - prevTick.bid) + MathAbs(tick.ask - prevTick.ask);
double
avgPrice = (tick.bid + tick.ask) / 2.0;
double
priceChangePct = priceChange / avgPrice;

// 2.
Spread
change
factor(widening
spread = more
volume)
double
currentSpread = tick.ask - tick.bid;
double
prevSpread = prevTick.ask - prevTick.bid;
double
spreadChange = MathAbs(currentSpread - prevSpread);

// 3.
Time
factor(faster
ticks = more
volume)
double
timeDiff = (tick.time_msc - prevTick.time_msc) / 1000.0; // seconds
double
timeFactor = timeDiff > 0 ? 1.0 / timeDiff: 1.0;
timeFactor = MathMin(timeFactor, 10.0); // Cap
at
10
x

// 4.
Calculate
final
volume
volume *= (1.0 + priceChangePct * 100.0); // Price
impact
volume *= (1.0 + spreadChange * 10.0); // Spread
impact
volume *= MathSqrt(timeFactor); // Time
impact(sqrt
to
dampen)

return volume;
}

// +------------------------------------------------------------------+
   // | Calculate
volume
metrics
for tick window |
// +------------------------------------------------------------------+
void CalculateVolumeMetrics(MqlTick & ticks[], int count,
double & avgVolume, double & totalVolume,
double & buyVolume, double & sellVolume,
double & vpin, double & volumeWeightedPrice)
{
avgVolume = 0;
totalVolume = 0;
buyVolume = 0;
sellVolume = 0;
double
priceVolumeSum = 0;

for (int i = 1; i < count; i++)
    {
        double
    volume = 0;

    // Use
    real
    volume if available, otherwise
    calculate
    if (ticks[i].volume_real > 0)
    {
        volume = ticks[i].volume_real;
    }
    else if (CalculateVolume)
    {
    volume = CalculateInferredVolume(ticks[i], ticks[i-1], VolumeMultiplier);
    }
    else
    {
    volume = VolumeMultiplier; // Fixed volume per tick
    }

    totalVolume += volume;

    // Determine if buy or sell based on price movement
    double midPrice = (ticks[i].bid + ticks[i].ask) / 2.0;
    double prevMidPrice = (ticks[i-1].bid + ticks[i-1].ask) / 2.0;

    if (midPrice > prevMidPrice)
    {
    buyVolume += volume;
    }
    else if (midPrice < prevMidPrice)
    {
    sellVolume += volume;
    }
    else
    {
    // No change - split volume
    buyVolume += volume / 2.0;
    sellVolume += volume / 2.0;
    }

    priceVolumeSum += midPrice * volume;
    }

avgVolume = count > 0 ? totalVolume / count: 0;

// Calculate
VPIN(Volume - Synchronized
Probability
of
Informed
Trading)
double
totalBuySell = buyVolume + sellVolume;
vpin = totalBuySell > 0 ? MathAbs(buyVolume - sellVolume) / totalBuySell: 0;

// Calculate
VWAP
volumeWeightedPrice = totalVolume > 0 ? priceVolumeSum / totalVolume: 0;
}

// +------------------------------------------------------------------+
   // | Export
tick
data |
// +------------------------------------------------------------------+
void
ExportTickData()
{
if (ExportAllSymbols)
    {
        ExportAllSymbolsTicks();
    }
    else
    {
        ExportCurrentSymbolTicks();
    }

    lastExportTime = TimeCurrent();
    }

    // +------------------------------------------------------------------+
       // | Export
    ticks
    for all symbols |
    // +------------------------------------------------------------------+
    void ExportAllSymbolsTicks()
    {
    int
    totalSymbols = SymbolsTotal(true);

    for (int i = 0; i < totalSymbols; i++)
        {
            string
        symbol = SymbolName(i, true);
        if (symbol != "")
        {
        ExportSymbolTicks(symbol);
        }
        }
    }

    // +------------------------------------------------------------------+
       // | Export
    ticks
    for current symbol |
    // +------------------------------------------------------------------+
    void ExportCurrentSymbolTicks()
    {
    ExportSymbolTicks(_Symbol);
    }

    // +------------------------------------------------------------------+
       // | Export
    ticks
    for specific symbol |
    // +------------------------------------------------------------------+
    void ExportSymbolTicks(string symbol)
    {
    MqlTick
    ticks[];

    // Calculate
    time
    range
    datetime
    currentTime = TimeCurrent();
    datetime
    fromTime = currentTime - (TicksLookbackMinutes * 60);

    // Copy
    ticks
    ulong
    from_msc = (ulong)
    fromTime * 1000;
    ulong
    to_msc = (ulong)
    currentTime * 1000;
    int
    copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, from_msc, to_msc);

    if (copied <= 0)
        {
            copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, MaxTicksPerExport);
        }

        if (copied <= 0)
            {
                Print("No ticks available for ", symbol);
        return;
        }

        // Calculate
        volume
        metrics
        double
        avgVolume, totalVolume, buyVolume, sellVolume, vpin, vwap;
        CalculateVolumeMetrics(ticks, copied, avgVolume, totalVolume, buyVolume, sellVolume, vpin, vwap);

        Print(symbol, ": Exporting ", copied, " ticks | Total Volume: ",
        DoubleToString(totalVolume, 2), " | VPIN: ", DoubleToString(vpin, 4));

        // Generate
        filename
        string
        filename = GetExportFilename(symbol);

        // Write
        data
        with volume metrics
        WriteEnhancedFormat(filename, symbol, ticks, copied,
        avgVolume, totalVolume, buyVolume, sellVolume, vpin, vwap);
        }

        // +------------------------------------------------------------------+
           // | Get
        export
        filename |
        // +------------------------------------------------------------------+
        string
        GetExportFilename(string
        symbol)
        {
            string
        extension = ".csv";
        string
        filename;

        if (OverwriteFile)
        {
        filename = ExportFolder + "\\" + symbol + "_ticks" + extension;
        }
        else
        {
        datetime
        now = TimeCurrent();
        string
        timestamp = TimeToString(now, TIME_DATE | TIME_SECONDS);
        StringReplace(timestamp, " ", "_");
        StringReplace(timestamp, ":", "");
        filename = ExportFolder + "\\" + symbol + "_ticks_" + timestamp + extension;
        }

        return filename;
        }

        // +------------------------------------------------------------------+
           // | Write
        enhanced
        format
        with volume calculations |
        // +------------------------------------------------------------------+
        void WriteEnhancedFormat(string filename, string symbol, MqlTick & ticks[], int count,
        double
        avgVolume, double
        totalVolume, double
        buyVolume,
        double
        sellVolume, double
        vpin, double
        vwap)
        {
        int
        handle = FileOpen(filename, FILE_WRITE | FILE_ANSI);

        if (handle == INVALID_HANDLE)
            {
                Print("Failed to open file: ", filename);
        return;
        }

        // Write
        data
        header
        directly(no
        metadata)
        FileWrite(handle,
                  "timestamp,timestamp_ms,bid,ask,last,spread,real_volume,inferred_volume,cum_volume,tick_direction,buy_pressure");

        // Initialize
        cumulative
        volume
        double
        cumVolume = 0;

        // Write
        tick
        data
        with volume calculations
        for (int i = count - 1; i >= 0; i--) // Newest first
        {
        datetime
        time = (datetime)(ticks[i].time_msc / 1000);
        double
        spread = ticks[i].ask - ticks[i].bid;

        // Calculate
        inferred
        volume
        double
        inferredVol = VolumeMultiplier;
        if (i > 0 & & CalculateVolume)
            {
                inferredVol = CalculateInferredVolume(ticks[i], ticks[i - 1], VolumeMultiplier);
            }

            // Use
            real
            volume if available
            double
            volume = ticks[i].volume_real > 0 ? ticks[i].volume_real: inferredVol;
            cumVolume += volume;

            // Determine
            tick
            direction
            string
            direction = "NEUTRAL";
            double
            buyPressure = 0.5;
            if (i > 0)
                {
                    double
                midPrice = (ticks[i].bid + ticks[i].ask) / 2.0;
                double
                prevMidPrice = (ticks[i - 1].bid + ticks[i - 1].ask) / 2.0;

                if (midPrice > prevMidPrice)
                {
                direction = "UP";
                buyPressure = 0.7 + 0.3 * (midPrice - prevMidPrice) / prevMidPrice;
                }
                else if (midPrice < prevMidPrice)
                {
                direction = "DOWN";
                buyPressure = 0.3 - 0.3 * (prevMidPrice - midPrice) / prevMidPrice;
                }

                // Clamp buy pressure between 0 and 1
                buyPressure = MathMax(0, MathMin(1, buyPressure));
                }

            string
            line = StringFormat("%s,%lld,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%s,%.3f",
                                TimeToString(time, TIME_DATE | TIME_SECONDS),
                                ticks[i].time_msc,
                                ticks[i].bid,
                                ticks[i].ask,
                                ticks[i].last,
                                spread,
                                ticks[i].volume_real,
                                inferredVol,
                                cumVolume,
                                direction,
                                buyPressure);

            FileWrite(handle, line);
            }

            FileClose(handle);
            Print("Exported ", count, " ticks with volume analysis to ", filename);
            }

            // +------------------------------------------------------------------+
               // | Create
            export
            directory |
            // +------------------------------------------------------------------+
            bool
            CreateExportDirectory()
            {
            if (!FolderCreate(ExportFolder))
            {
            int
            error = GetLastError();
            if (error != 5019) // 5019 = folder already exists
            {
                Print("Failed to create folder: ", error);
            return false;
            }
            }
            return true;
            }

            // +------------------------------------------------------------------+
               // | Expert
            deinitialization
            function |
            // +------------------------------------------------------------------+
            void
            OnDeinit(const
            int
            reason)
            {
            EventKillTimer();
            Print("TickDataExporter stopped");
            }

            // +------------------------------------------------------------------+
               // | Timer
            function |
            // +------------------------------------------------------------------+
            void
            OnTimer()
            {
            if (EnableAutoExport)
                {
                    ExportTickData();
                }
                }

                // +------------------------------------------------------------------+
                   // | ChartEvent
                function
                for manual control |
                // +------------------------------------------------------------------+
                void OnChartEvent(const int id,
                const
                long & lparam,
                const
                double & dparam,
                const
                string & sparam)
                {
                // Press
                'E'
                to
                trigger
                manual
                export
                if (id == CHARTEVENT_KEYDOWN & & lparam == 69) // 69 = 'E' key
                {
                    Print("Manual export triggered");
                ExportTickData();
                }

                // Press
                'V'
                for volume analysis
                    if (id == CHARTEVENT_KEYDOWN & & lparam == 86) // 86 = 'V' key
                {
                    AnalyzeRecentVolume(_Symbol);
                }
                }

                // +------------------------------------------------------------------+
                   // | Analyze
                recent
                volume
                patterns |
                // +------------------------------------------------------------------+
                void
                AnalyzeRecentVolume(string
                symbol)
                {
                MqlTick
                ticks[];
                int
                copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 10000);

                if (copied > 0)
                    {
                        double
                    avgVol, totalVol, buyVol, sellVol, vpin, vwap;
                    CalculateVolumeMetrics(ticks, copied, avgVol, totalVol, buyVol, sellVol, vpin, vwap);

                    Print("=== Volume Analysis for ", symbol, " ===");
                    Print("Ticks analyzed: ", copied);
                    Print("Average Volume/Tick: ", DoubleToString(avgVol, 2));
                    Print("Total Volume: ", DoubleToString(totalVol, 2));
                    Print("Buy Volume: ", DoubleToString(buyVol, 2), " (",
                          DoubleToString(buyVol / totalVol * 100, 1), "%)");
                    Print("Sell Volume: ", DoubleToString(sellVol, 2), " (",
                          DoubleToString(sellVol / totalVol * 100, 1), "%)");
                    Print("VPIN (Toxicity): ", DoubleToString(vpin, 4));
                    Print("VWAP: ", DoubleToString(vwap, 5));

                    // Check if real
                    volume is available
                    bool
                    hasRealVolume = false;
                    for (int i = 0; i < copied & & i < 100; i++)
                    {
                    if (ticks[i].volume_real > 0)
                    {
                    hasRealVolume = true;
                break;
                }
                }
                Print("Real Volume Available: ", hasRealVolume ? "YES": "NO (using inferred)");
                }
                }