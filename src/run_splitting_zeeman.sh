#!/usr/bin/env bash

if [[ -e "../data/$1.csv" ]]; then
    root -l -e ".L splitting_zeeman.cpp" -e "splitting_zeeman(\"../data/$1.csv\")"
else
    echo "utilizzo: $(basename $0) <fname> "
    echo "Il file utilizzato sarà ../data/<fname>.csv"
fi
