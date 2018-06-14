@echo off

echo running with options=%1

if not %1_==--clean_ goto build

echo cleaning...

del /s /q .\doc\generated\*
del /s /q .\build\sphinx\doctrees\*
del /s /q .\build\sphinx\html\*
del /s /q .\build\sphinx\latex\*

:build

echo building...

python setup.py build_sphinx --builder html,latex

cd .\build\sphinx\latex
texify --clean --pdf sdepy_manual.tex

cd ..\..\..


