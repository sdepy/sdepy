#!/bin/bash

echo running with options=$@

if [[ $1 == '--clean' && -f ./sdepy/__init__.py ]]; then

    echo cleaning...
    rm -fr ./doc/generated/*
    rm -fr ./build/sphinx/doctrees/*
    rm -fr ./build/sphinx/html/*
    rm -fr ./build/sphinx/latex/*

fi

if [[ $1 == '--html' ]]; then

    echo building html...
    python setup.py build_sphinx --builder html

fi

if [[ $1 == '--all' ]]; then

    echo building html and latex...
    python setup.py build_sphinx --builder html,latex

    cd ./build/sphinx/latex
    texify --clean --pdf sdepy_manual.tex
    cd ../../..
    mkdir -p ./dist
    mv ./build/sphinx/latex/sdepy_manual.pdf ./dist

fi
