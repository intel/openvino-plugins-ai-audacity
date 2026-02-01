call env.bat
call %OPENVINO_GENAI_DIR%\setupvars.bat
set LIBTORCH_ROOTDIR=%LIBTORCH_DIR%
set Path=%LIBTORCH_ROOTDIR%\lib;%Path%
