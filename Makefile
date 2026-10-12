all: tables

tables:
	python3 python/joe.py exact
	python3 python/joe.py listed
	python3 python/joe.py extra
	python3 python/joe.py sens
	python3 python/joe.py mc
	python3 python/joe.py figures
	python3 python/joe_tables.py

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
