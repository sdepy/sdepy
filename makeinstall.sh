#!/bin/bash

echo running with options=$@

rm -f ./dist/sdepy*.whl
rm -f ./dist/sdepy*.gz
python setup.py sdist bdist_wheel
pip install --no-deps --ignore-installed ./dist/sdepy*.whl

if [[ $1 == '-q' || $1 == '--quickguide' ]]; then
    python -c 'from tools import *; quickguide_make()'
fi

