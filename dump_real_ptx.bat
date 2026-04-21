@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 exit /b 1

set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA%\bin;%PATH%"
cd /d "%~dp0"

if exist build_ptx rmdir /s /q build_ptx
mkdir build_ptx

"%CUDA%\bin\nvcc.exe" -arch=sm_120 -O3 --use_fast_math ^
    -std=c++17 -allow-unsupported-compiler ^
    -DTLLM_QKV_HAND_TUNED_PTX ^
    -I include ^
    --keep --keep-dir build_ptx ^
    -dc src/kernels/qkv_proj.cu ^
    -o build_ptx/qkv_proj.obj 2>&1

exit /b %errorlevel%
