@echo off
set "PYTHONHOME="
set "PYTHONPATH="
"%~dp0python\python.exe" "%~dp0bootstrap.py"
if errorlevel 1 pause
