@echo off
REM Daily Power Ratings Update Script for Windows Task Scheduler
REM
REM To schedule:
REM 1. Open Task Scheduler (taskschd.msc)
REM 2. Create Basic Task
REM 3. Set trigger: Daily at 6:00 AM
REM 4. Action: Start a program
REM 5. Program: C:\Users\mpgra\Documents\cbb_power_rating\update_ratings.bat
REM 6. Start in: C:\Users\mpgra\Documents\cbb_power_rating

cd /d "C:\Users\mpgra\Documents\cbb_power_rating"

REM Log start time
echo [%date% %time%] Starting ratings update >> logs\update.log

REM Run the update script
python update_ratings.py >> logs\update.log 2>&1

REM Log completion
echo [%date% %time%] Update completed >> logs\update.log
echo. >> logs\update.log
