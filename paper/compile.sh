#!/bin/bash
# Compile the LaTeX paper to PDF

echo "Compiling emotion_recognition_paper.tex..."

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode emotion_recognition_paper.tex
pdflatex -interaction=nonstopmode emotion_recognition_paper.tex

# Clean auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo ""
echo "Done! Output: emotion_recognition_paper.pdf"
