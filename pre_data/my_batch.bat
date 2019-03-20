@echo off
if "%1"=="" goto HELP
if not exist "%1" goto HELP


set DELAY=10
set CWD=%CD%
set VMRUN="E:\VMware\vmrun.exe"
set VMX="D:\ÐéÄâ»ú\Win10\Windows 10 x64.vmx"
set VM_SNAPSHOT="real"
SET VM_USER="msi"
set VM_PASS="123456"
set FILENAME=%~nx1
set SCRIPT_PATH="C:\Users\msi\Desktop\my_sandbox_script.py"
set LOG_PATH="C:\Malware"
set SAVE_PATH=%2
set ZIP_PATH="C:\Tools\zip.exe"


%VMRUN% -T ws revertToSnapshot %VMX% %VM_SNAPSHOT%
%VMRUN% -T ws start %VMX% nogui
%VMRUN% -gu %VM_USER%  -gp %VM_PASS% copyFileFromHostToGuest %VMX% "%1" C:\Malware\%FILENAME%
%VMRUN% -T ws -gu %VM_USER% -gp %VM_PASS% runProgramInGuest %VMX% "C:\Users\msi\AppData\Local\Programs\Python\Python35\python.exe" %SCRIPT_PATH% -t %DELAY% -f %FILENAME%
if %ERRORLEVEL%==1 goto ERROR1
:%VMRUN% -T ws -gu %VM_USER% -gp %VM_PASS% runProgramInGuest %VMX% %ZIP_PATH% -j C:\NoribenReports.zip %LOG_PATH%\*.*
%VMRUN% -gu %VM_USER%  -gp %VM_PASS% copyFileFromGuestToHost %VMX% %LOG_PATH%\%FILENAME%.yuri %SAVE_PATH%\%FILENAME%.yuri
goto END1

:ERROR1
echo [!] File did not execute in VM correctly.
goto END2

:HELP
echo Please provide executable filename as an argument.
echo For example:
echo %~nx0 C:\Malware\ef8188aa1dfa2ab07af527bab6c8baf7
goto END2

:END1
echo %1 completed!

:END2