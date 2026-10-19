Unicode True
!include "MUI2.nsh"
!include "x64.nsh"
Name "oollim"
!define MUI_ICON "stage\app\assets\oollim.ico"
!define MUI_UNICON "stage\app\assets\oollim.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "stage\app\assets\installer-welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "stage\app\assets\installer-header.bmp"
Icon "stage\app\assets\oollim.ico"
UninstallIcon "stage\app\assets\oollim.ico"
OutFile "oollim-setup.exe"
InstallDir "$PROFILE\oollim\launcher"
RequestExecutionLevel user
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "stage\app\LICENSE.md"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "oollim requires 64-bit Windows on an AMD64 processor."
    Abort
  ${EndIf}
FunctionEnd
Section "Install"
  SetOutPath "$INSTDIR"
  File /r "stage\*"
  CreateShortcut "$SMPROGRAMS\oollim.lnk" "$INSTDIR\launch.cmd" "" "$INSTDIR\app\assets\oollim.ico"
  CreateShortcut "$DESKTOP\oollim.lnk" "$INSTDIR\launch.cmd" "" "$INSTDIR\app\assets\oollim.ico"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\oollim" "DisplayName" "oollim"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\oollim" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  MessageBox MB_OK "Installed. Open oollim from Start to download dependencies and models and launch the dashboard. Internet access and approximately 10 GB plus temporary space are needed." /SD IDOK
SectionEnd
Section "Uninstall"
  Delete "$SMPROGRAMS\oollim.lnk"
  Delete "$DESKTOP\oollim.lnk"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  Delete "$INSTDIR\bootstrap.py"
  Delete "$INSTDIR\payload.json"
  Delete "$INSTDIR\launch.cmd"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\oollim"
  MessageBox MB_OK "Launcher removed. Downloaded models, environments and results are preserved in your user folders." /SD IDOK
SectionEnd
