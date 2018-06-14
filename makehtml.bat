@echo off

echo running with options=%1

if not %1_==--clean_ goto build

echo cleaning...

rmdir /s /q .\doc\generated
rmdir /s /q .\build\sphinx\doctrees
rmdir /s /q .\build\sphinx\html

:build

echo building...

python setup.py build_sphinx --builder html



