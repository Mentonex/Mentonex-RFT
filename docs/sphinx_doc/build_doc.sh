#!/bin/bash
sphinx-apidoc -f -o source ../../trinity -t _templates
make clean html
