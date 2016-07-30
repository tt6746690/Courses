

Compile latex

```
pdflatex template.tex
```

set up bibtex with biber

```tex
%% using bibtex with biber
\usepackage[
    backend=biber,
    style=authoryear-icomp,
    sortlocale=de_DE,
    natbib=true,
    url=false,
    doi=true,
    eprint=false
]{biblatex}
\addbibresource{reference.bib}
```


moving .bib to current directory
```bash
mv ~/Downloads/Your\ bibliography.bib ~/Desktop/2016\ summer/DONE\ LAB/ki67Paper/
mv -f ~/Desktop/My\ Collection.bib /Users/markwang/github/courseProjects/doneLab2016/ki67Paper/manuscript/reference.bib

```


---

Resources

[_Elsevier template_](https://www.elsevier.com/__data/assets/pdf_file/0008/56843/elsdoc-1.pdf)
[_bibtex doc_](http://ctan.mirror.rafal.ca/macros/latex/contrib/biblatex/doc/biblatex.pdf)
[_comparing latex packages: natlib vs. biblatex, and processing programs: bibTex vs biber_](http://tex.stackexchange.com/questions/25701/bibtex-vs-biber-and-biblatex-vs-natbib)  
[_CitethisforMe for getting .bib from the web_](http://www.citethisforme.com/)    
[_ICC R package_](https://cran.r-project.org/web/packages/ICC/ICC.pdf)  
[_A tutorial on computing ICC with R_](http://aliquote.org/memos/2011/04/29/computing-intraclass-correlation-with-r)  
[_IRR R package_](https://cran.r-project.org/web/packages/irr/irr.pdf)  
[_useful tutorial on computing ICC with irr package_](http://www.cookbook-r.com/Statistical_analysis/Inter-rater_reliability/)  
[_A good post on difference in kappa and ICC_](http://stats.stackexchange.com/questions/3539/inter-rater-reliability-for-ordinal-or-interval-data?noredirect=1&lq=1)
