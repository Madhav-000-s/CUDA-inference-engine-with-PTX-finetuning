@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 exit /b 1

set "CMAKE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d "%~dp0"
"%CMAKE%" --build build --target qkv_ptx
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build build --target qkv_ptx_tuned
exit /b %errorlevel%
