#!/bin/bash
BASE="${1%.*}"
pdflatex $BASE.tex
if [ $? -ne 0 ]; then
  echo "comp errr"
  exit 1
fi
bibtex $BASE
pdflatex $BASE.tex
pdflatex $BASE.tex

#okular main.pdf
exit 0
