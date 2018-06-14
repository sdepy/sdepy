@echo off

del .\dist\sdepy*.whl
del .\dist\sdepy*.gz

python -c "from tools import *; quickguide_make()"

python setup.py sdist bdist_wheel

for %%w in (.\dist\sdepy*.whl) do pip install --no-deps --ignore-installed %%w
