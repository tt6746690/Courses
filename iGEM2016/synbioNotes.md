

#### BioBrick  

[__Idempotent Vector Design for Standard Assembly of Biobricks__](http://dspace.mit.edu/bitstream/handle/1721.1/21168/biobricks.pdf?sequence=1)  

1. The BioBrick assembly standard enables the distributed production of a collection of compatible biological parts
2. Since engineers carry out the exact same operation every time that they want to combine two BioBrick parts, the assembly process is amenable to optimization and automation, in contrast to more traditional _ad hoc_ molecular cloning approaches.

[_The Registry_](http://parts.igem.org/Main_Page)  

[_Team Search_](http://igem-qsf.github.io/iGEM-Team-Seeker/dist/)






#### Synthetic Biology Softwares  

[__GenoLIB: a database of biological parts derived from a library of common plasmid features__](http://nar.oxfordjournals.org/content/43/10/4823.full)   
+ use Synthetic Biology Open Language (SBOL)
+ ~2000 annotated plasmids
+ better curated than registry

[__The second wave of synthetic biology: from modules to systems__](http://getit.library.utoronto.ca.myaccess.library.utoronto.ca/index.php/oneclick?sid=Entrez:PubMed&id=pmid:19461664)


how julian did wiki 2015  

starts with this markdown https://github.com/igemuoftATG/wiki2015/blob/master/src/markdown/Description.md

used in this handlebars template (see handlebarsjs.com)
https://github.com/igemuoftATG/wiki2015/blob/master/src/Description.hbs

which makes use of the custom "markdown" handlebars "helper"
https://github.com/igemuoftATG/wiki2015/blob/master/helpers.coffee#L253

which puts this inside a <ul class="nav">:
marked(toc(handlebarsedMarkdown, {firsth1: false, maxdepth: 5}).content).slice(4)
(marked is a fucntion that takes markdown and gives html, toc comes from require('markdown-toc')

notice how h1 is ignored (firsth1: false). otherwise all items below the first indented by 1

then start scrollspy on #tableofcontents:
https://github.com/igemuoftATG/wiki2015/blob/master/src/lib/main.js#L5

which basically will attach an "active" css class to the proper <ul> and <li> dom elements in the .tableofcontents which I then did a bunch of CSS styling for here, it was kinda annoying, had to style like ul, li, ul, li, ul, li to handle 3 deep and different colours grin emoticon
(see li.active, ul.active, etc.)
https://github.com/igemuoftATG/wiki2015/blob/master/src/sass/_main.scss#L125

its a bootstrap plugin that basically "just works" if you load in the bootsrap js/css and conform your html to the bootstrap classes:
http://getbootstrap.com/javascript/#scrollspy
