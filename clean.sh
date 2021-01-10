#!/bin/bash

echo running with options=$@

if [[ $1 = '--all' && -f ./sdepy/__init__.py ]]; then

    echo cleaning...

    rm -fr ./doc/generated
    rm -fr ./build
    rm -fr ./sdepy.egg-info

    rm -f ./sdepy/tests/cfr/*.png
    rm -f ./sdepy/tests/cfr/*err_realized.txt

fi
