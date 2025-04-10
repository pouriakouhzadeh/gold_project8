//+------------------------------------------------------------------+
//|                                                       Expert.mq4 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
int FindExecutableW(string lpFile, string lpDirectory);
#import

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string   An              = "NULL";
string   An_position     = "NULL";
string   An_proba        = "NULL";
datetime LastBarM30      = 0;
datetime lastBarTime_M30 = 0;
bool     newBarM30       = false;
int      ticket;
datetime brokerTime_25_55= 0;
int minute_25_55         = 0;
input int Minimum_Spread = 40;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(1);
   newBarM30 = false;
   LastBarM30    = 0;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   /*   if(IsNewBar())
        {
         Print("New bar .............................");
         close_all_positions();
        }
   */
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
   if(working_expert() == true)
     {
      close_all_positions();
      My_print("working_expert","Expert is off untill 01:30 AM", 150);
      My_print("position","---",40);
      My_print("proba","---",100);
      Print("working_expert","Expert is off untill 01:30 AM");
      return;
     }

   static datetime lastCloseTime = 0; // زمان آخرین بستن پوزیشن‌ها

   int timeframe = PeriodSeconds(); // مدت‌زمان هر کندل بر حسب ثانیه
   datetime currentTime = TimeCurrent(); // زمان جاری سرور
   datetime candleOpenTime = iTime(Symbol(), 0, 0); // زمان باز شدن کندل جاری
   int secondsToClose = timeframe - (int)MathFloor(currentTime - candleOpenTime);

// اگر 1 یا 2 ثانیه قبل از بسته شدن کندل است و قبلاً اجرا نشده است
   if((secondsToClose == 1 || secondsToClose == 2) && lastCloseTime != candleOpenTime)
     {
      Print("Closing positions before candle close... (", secondsToClose, " seconds left)");
      close_all_positions(); // اجرای تابع بستن تمام پوزیشن‌ها
      lastCloseTime = candleOpenTime; // جلوگیری از اجرای مجدد در همان کندل
     }




   Comment("Spread = ",MarketInfo(Symbol(), MODE_SPREAD));
   if(newBarM30 == false)
      newBarM30 = CheckNewBarM30(30, LastBarM30);
   Print("newBarM30 = ", newBarM30);
   if(newBarM30)
     {
      //DeleteCommonFile();
      close_all_positions();
      get_answer_file();
      answer_read();
      get_position();
     }

   int handle = FileOpen("Answer.txt", FILE_READ | FILE_COMMON|FILE_ANSI);

   if(handle != INVALID_HANDLE)  // یا if (handle >= 0)
     {
      if(IsPositionOpen(Symbol()))
        {
         FileClose(handle);
         DeleteCommonFile();
        }
      FileClose(handle);
     }

   brokerTime_25_55 = TimeCurrent();
   minute_25_55 = TimeMinute(brokerTime_25_55);
   if((minute_25_55 > 25 && minute_25_55 < 29) || (minute_25_55 > 55 && minute_25_55 < 59))
     {
      DeleteCommonFile();
      newBarM30 = false;
     }
//Print("An_proba = ", StringToDouble(StringSubstr(An_proba,0,3)));
  }
//+------------------------------------------------------------------+
void answer_read()
  {
   int fileHandle = FileOpen("Answer.txt", FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(fileHandle!=INVALID_HANDLE)
     {
      An = FileReadString(fileHandle, 10);
      An_position = StringSubstr(An,0,3);
      An_proba = StringSubstr(An,4,9);
      //Print("An_position = ", An_position);

     }

   FileClose(fileHandle);
   My_print("position",An_position,40);
   My_print("proba",An_proba,100);

  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void get_answer_file()
  {
   string batFilePath = "C://Users//Forex//AppData//Roaming//MetaQuotes//Terminal//Common//Files//Get_answer_file.bat";
   int result_get_answer = ShellExecuteW(0, "runas", "cmd.exe", "/c " + batFilePath, NULL, 0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void send_acknowledge_file()
  {
   string batFilePath = "C://Users//Forex//AppData//Roaming//MetaQuotes//Terminal//Common//Files//Send_acknowledge_file.bat";
   int result_send_acn = ShellExecuteW(0, "runas", "cmd.exe", "/c " + batFilePath, NULL, 0);
  }
//+------------------------------------------------------------------+
bool IsPositionOpen(string symbol)
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == symbol && (OrderType() == OP_BUY || OrderType() == OP_SELL))
           {
            return true;
           }
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
void get_position()
  {
   Print("Try to get position ...");
   double Lot = 0;
   if(An_position == "BUY")
     {
      if(spread_calc(Symbol(), Minimum_Spread))
        {
         if(!IsPositionOpen(Symbol()))
           {
            if(check_price("BUY")== false)
              {
               Lot = Lot_calc("BUY");
               if(((int)(Lot * 10) % 2) != 0)  // استفاده از % به جای mode
                 {
                  Lot = ((((Lot_calc("BUY") * 10)-1) / 10) / 2); // اصلاح فرمول و ذخیره نتیجه
                 }
               else
                 {
                  Lot = Lot_calc("BUY") /2;
                 }
               ticket = OrderSend(Symbol(),OP_BUY,Lot,Ask,10,0,0,"Gold",1000,0,Blue);
               if(ticket >= 1)
                 {
                  PlaceLimitOrder("BUY",Lot,15);
                  An = "---";
                  An_position = "---";
                  An_proba = "---";
                  send_acknowledge_file();
                  newBarM30 = false;
                  if(working_expert())
                    {
                     My_print("working_expert","", 150);
                    }
                 }
               if(ticket < 0)
                 {
                  Print("Error in OrderSend: ", GetLastError());
                 }
              }
            else
              {
               Lot = Lot_calc("BUY");
               ticket = OrderSend(Symbol(),OP_BUY,Lot,Ask,10,0,0,"Gold",1000,0,Blue);;
               if(ticket >= 1)
                 {
                  An = "---";
                  An_position = "---";
                  An_proba = "---";
                  send_acknowledge_file();
                  newBarM30 = false;
                  if(working_expert())
                    {
                     My_print("working_expert","", 150);
                    }
                 }
               if(ticket < 0)
                 {
                  Print("Error in OrderSend: ", GetLastError());
                 }

              }

           }
        }
     }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
   if(An_position == "SEL")
     {
      if(spread_calc(Symbol(), Minimum_Spread))
        {
         if(!IsPositionOpen(Symbol()))
           {
            if(check_price("SELL") == false)
              {
               Lot = Lot_calc("SELL");
               if(((int)(Lot * 10) % 2) != 0)  // استفاده از % به جای mode
                 {
                  Lot = ((((Lot_calc("SELL") * 10)-1) / 10) / 2); // اصلاح فرمول و ذخیره نتیجه
                 }
               else
                 {
                  Lot = Lot_calc("SELL") /2;
                 }
               ticket = OrderSend(Symbol(),OP_SELL,Lot,Bid,10,0,0,"Gold", 60, 0,Red);
               if(ticket >= 1)
                 {
                  PlaceLimitOrder("SELL",Lot,15);
                  An = "---";
                  An_position = "---";
                  An_proba = "---";
                  send_acknowledge_file();
                  newBarM30 = false;
                  if(working_expert())
                    {
                     My_print("working_expert","", 150);
                    }
                 }
               if(ticket < 0)
                 {
                  Print("Error in OrderSend: ", GetLastError());
                 }
              }
            else
              {
               Lot = Lot_calc("SELL");
               ticket = OrderSend(Symbol(),OP_SELL,Lot,Bid,10,0,0,"Gold", 60, 0,Red);
               if(ticket >= 1)
                 {
                  An = "---";
                  An_position = "---";
                  An_proba = "---";
                  send_acknowledge_file();
                  newBarM30 = false;
                  if(working_expert())
                    {
                     My_print("working_expert","", 150);
                    }
                 }
               if(ticket < 0)
                 {
                  Print("Error in OrderSend: ", GetLastError());
                 }

              }
           }
        }
     }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
   if(An_position == "NAN")
     {
      An = "---";
      An_position = "---";
      An_proba = "---";
      send_acknowledge_file();
      newBarM30 = false;
      if(working_expert())
        {
         My_print("working_expert","", 150);
        }
     }

  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool spread_calc(string symbol, int amount)
  {
   Print("Spread => ",MarketInfo(symbol, MODE_SPREAD));
   if(MarketInfo(symbol, MODE_SPREAD) <= amount)
     {
      return true ;
     }
   return false ;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
void close_all_positions()
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))  // بررسی موفقیت انتخاب سفارش
        {
         int orderType = OrderType();
         int orderTicket = OrderTicket();
         double orderLots = OrderLots();

         // برای بستن پوزیشن‌های باز (BUY, SELL)
         if(orderType == OP_BUY || orderType == OP_SELL)
           {
            double closePrice = (orderType == OP_BUY) ? Bid : Ask; // قیمت مناسب برای بستن سفارش
            bool orderClosed = OrderClose(orderTicket, orderLots, closePrice, 6, clrNONE);

            if(orderClosed)
              {
               Print("Closed position: ", orderTicket, " at price: ", closePrice);
              }
            else
              {
               Print("Failed to close position: ", orderTicket, " Error: ", GetLastError());
              }
           }

         // برای حذف سفارش‌های معلق (BUY_LIMIT, SELL_LIMIT)
         else
            if(orderType == OP_BUYLIMIT || orderType == OP_SELLLIMIT)
              {
               bool orderDeleted = OrderDelete(orderTicket);

               if(orderDeleted)
                 {
                  Print("Deleted pending order: ", orderTicket);
                 }
               else
                 {
                  Print("Failed to delete pending order: ", orderTicket, " Error: ", GetLastError());
                 }
              }
        }
      else
        {
         Print("Error selecting order at position: ", i, " Error: ", GetLastError());
        }
     }
   Print("All positions and pending orders processed.");
  }


//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void DeleteCommonFile()
  {
   string batFilePath = "C://Users//Forex//AppData//Roaming//MetaQuotes//Terminal//Common//Files//DeleteAnswer.bat";
   int result_get_answer = ShellExecuteW(0, "runas", "cmd.exe", "/c " + batFilePath, NULL, 0);
  }

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool CheckNewBarM30(int tf, datetime lastBarTime)
  {
   datetime t = iTime(Symbol(), tf, 0);
   long volume = iVolume(Symbol(), tf, 0);
   if(t <= 0 || volume == 0)
      return false;
   if(lastBarTime == 0)
     {
      LastBarM30 = t;
      return false;
     }
   if(t != lastBarTime)
     {
      LastBarM30 = t;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
bool check_price(string position)
  {
   if(position == "BUY")
     {
      if(iClose(Symbol(),0,1) >= Bid)
        {
         return true;
        }
     }
   if(position == "SELL")
     {
      if(iClose(Symbol(),0,1) <= Bid)
        {
         return true;
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
double Lot_calc(string position)
  {
   if(position == "BUY")
     {
      return (((round(StrToDouble(An_proba) * 10)) / 10) * 2);
     }
   if(position == "SELL")
     {
      return (((round((1 - (StrToDouble(An_proba)))  * 10)) /10) * 2) ;
     }
   return 0.0;
  }
//+------------------------------------------------------------------+
void My_print(string name, string text, int X)
  {
// بررسی کنید که آیا شیء وجود دارد یا نه
   if(ObjectFind(name) < 0) // شیء پیدا نشد، پس ایجادش کن
     {
      if(!ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0))
        {
         Print("خطا در ایجاد آبجکت: ", name, " خطای: ", GetLastError());
         return;
        }
     }

// تنظیم ویژگی‌های شیء متنی
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_CORNER, 10);       // گوشه بالا چپ
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 60);   // فاصله از لبه
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, X);   // فاصله از بالا
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrYellow);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 24);    // اندازه فونت
   ObjectSetString(0, name, OBJPROP_FONT, "Tahoma");   // فونت
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, false);   // نمایش دادن شیء

// به‌روز‌رسانی نمودار برای نمایش تغییرات
   ChartRedraw();
  }
//+------------------------------------------------------------------+
bool working_expert()
  {
   datetime brokerTime = TimeCurrent(); // زمان جاری بروکر
   int hour = TimeHour(brokerTime);     // استخراج ساعت از زمان بروکر
   int minute = TimeMinute(brokerTime); // استخراج دقیقه از زمان بروکر

   if((hour == 22 && minute >= 30) ||  // از 22:30 تا 23:59
      (hour == 23) ||                 // کل ساعت 23
      (hour == 0) ||                  // کل ساعت 00:00 تا 00:59
      (hour == 1 && minute < 30))     // از 01:00 تا 01:30
     {
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
void PlaceLimitOrder(string position, double lot, int maxAttempts)
  {
   double entryPrice;
   int attempt = 0;
   int order_ticket;

   if(position == "BUY")  // سفارش BUY_LIMIT
     {
      entryPrice = iClose(Symbol(), 0, 1); // قیمت ورود اولیه
     }
   else
      if(position == "SELL")  // سفارش SELL_LIMIT
        {
         entryPrice = iClose(Symbol(), 0, 1); // قیمت ورود اولیه
        }
      else
        {
         Print("Error: Invalid position type. Use 'BUY' or 'SELL'.");
         return;
        }

   while(attempt < maxAttempts)
     {
      if(position == "BUY")
        {
         order_ticket = OrderSend(Symbol(), OP_BUYLIMIT, lot, entryPrice, 3, 0, 0, "Buy Limit Order", 0, 0, clrAqua);
        }
      else // position == "SELL"
        {
         order_ticket = OrderSend(Symbol(), OP_SELLLIMIT, lot, entryPrice, 3, 0, 0, "Sell Limit Order", 0, 0, clrRed);
        }

      if(order_ticket > 0)  // سفارش با موفقیت ثبت شد
        {
         Print(position, "_LIMIT open in price: ", entryPrice);
         break;
        }
      else // سفارش رد شد، پس قیمت ورود را تنظیم می‌کنیم
        {
         int errorCode = GetLastError();
         Print("Error in ", position, ": ", errorCode);

         if(errorCode == 130)  // خطای اسپرد (Invalid Stops)
           {
            if(position == "BUY")
               entryPrice -= 1 * Point; // کاهش 1 پیپت برای BUY_LIMIT
            else
               entryPrice += 1 * Point; // افزایش 1 پیپت برای SELL_LIMIT

            attempt++;
            Print("Adjusting price due to spread: ", entryPrice);
            Sleep(500); // تأخیر کوتاه قبل از تلاش مجدد
           }
         else // اگر خطای دیگری رخ داد، خروج
           {
            break;
           }
        }
     }

   if(attempt >= maxAttempts)
     {
      Print("Can not place ", position, "_LIMIT order.");
     }
  }
//+------------------------------------------------------------------+
