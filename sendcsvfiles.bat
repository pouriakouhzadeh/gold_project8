@echo off
cd /d "C:\Users\Forex\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
pscp.exe -pw P1755063881k C:\Users\Forex\AppData\Roaming\MetaQuotes\Terminal\Common\Files\*.csv pouria@192.168.12.10:/home/pouria/gold_project3/
del *.csv
exit