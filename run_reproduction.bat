@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "TASK_ROOT=%CD%"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TASK_ROOT%\scripts\bootstrap_windows.ps1"
if errorlevel 1 goto failed
"%TASK_ROOT%\.runtime\windows-x86_64\python\python.exe" -I -B -X utf8 "%TASK_ROOT%\scripts\runtime.py" %*
set "TASK_STATUS=%ERRORLEVEL%"
if not "%TASK_STATUS%"=="0" if "%~1"=="" pause
popd
exit /b %TASK_STATUS%
:failed
echo 环境准备没有完成。请检查网络及文件夹写入权限，然后重试。
if "%~1"=="" pause
popd
exit /b 2
