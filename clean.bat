@echo off

echo running with options=%1

copy .\build\sphinx\latex\sdepy_manual.pdf .

if not %1_==--all_ goto exit

echo cleaning...

del /s /q .\doc\generated\*

del /s /q .\build\sphinx\doctrees\*
del /s /q .\build\sphinx\html\*
del /s /q .\build\sphinx\latex\*
del /s /q .\build\lib\*

del /s /q .\sdepy\tests\cfr\*.png
del /s /q .\sdepy\tests\cfr\*err_realized.txt

:exit
