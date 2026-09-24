all: replicate

replicate:
	python3 python/replicate.py

paper.pdf: paper.tex refs.bib
	pdflatex -interaction=nonstopmode paper
	bibtex paper
	pdflatex -interaction=nonstopmode paper
	pdflatex -interaction=nonstopmode paper

push:
	git push origin HEAD

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc paper.spl
