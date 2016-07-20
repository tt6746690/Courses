

#### Experimental details

_Ki-67 project_  

KI67 assessment on sections cut from 6 individual TMA blocks of _invasive carcinoma_ consisting of 278 patients (GeparTrio breast cancer study). Each patient cases is represented by 3 0.6mm cores. Slides were stained with KI-67 antibody (Dako clone MIB-1) and scanned.
+ Manual count of _percentage_ of the invasive carcinoma cells showing positive nuclear staining (%cell stained per 100 cancer cells in the context of 3 cores); this serves as benchmark as the scoring correlates to clinical outcome.
+ Same 6 slides annotated on Aperio by 2 observers (inter-observer: Ade + Tian). The annotated regions were analyzed for Ki-67 positivity with _Aperio ePathology_  

(_Pearson correlation_ calculated with paired data from computer and manual scoring
    and _heteroscedastic T-test_ used to determine that the 2 observers that annotated the slides were not significantly different.)

---

_PAPER: _[__An International Ki67 Reproducibility Study__](http://jnci.oxfordjournals.org/content/105/24/1897.long)   

_Abstract_   
__Eight__ laboratories received __100__ breast cancer cases arranged into 1-mm core tissue microarrays-one set stained by the participating laboratory and one set stained by the central laboratory, both using antibody MIB-1. Each laboratory scored Ki67 as __percentage of positively stained invasive tumor cells__ using its own method. Six laboratories repeated scoring of 50 locally stained cases on 3 different days. Sources of variation were analyzed using random effects models with log2-transformed measurements. __Reproducibility__ was quantified by __intraclass correlation coefficient__ (ICC), and the approximate two-sided 95% confidence intervals (CIs) for the true intraclass correlation coefficients in these experiments were provided.

_Result_  
Intralaboratory reproducibility was high (ICC = 0.94; 95% CI = 0.93 to 0.97). Interlaboratory reproducibility was only moderate (central staining: ICC = 0.71, 95% CI = 0.47 to 0.78; local staining: ICC = 0.59, 95% CI = 0.37 to 0.68). Supplementary statistics [_here_](https://docs.google.com/viewer?url=http%3A%2F%2Fjnci.oxfordjournals.org%2Fcontent%2Fsuppl%2F2013%2F09%2F27%2Fdjt306.DC1%2Fjnci_JNCI_13_0538_s01.docx)  

_POINTS_

_Intralaboratory reproducibility_   
Plotted using [__Bland–Altman plots__](https://www-users.york.ac.uk/~mb55/meas/ba.pdf), which graph the difference in Ki67 between any two paired observations against their mean  

![bland-altman plot](http://jnci.oxfordjournals.org/content/105/24/1897/F1.large.jpg)

[C,D,E,H achieved high internal consistency] The __y-axis__ (difference in Ki67) represents the Ki67 value (percentage of positively stained tumor cells) of the former day minus that of the latter day. The __middle dashed line__ represents the average of the differences across all observations. Hence, a middle dashed line greater than 0 would indicate that the average Ki67 value of the former day is greater than that of the latter day, and vice versa.


_Interlaboratory Reproducibility_  

+ _Central Staining, Local Scoring Method_  
Table 2 provides, for each laboratory, __summary statistics__ of log2-transformed Ki67 scores observed when all laboratories used their own local scoring methods on centrally stained sections. All measurement data were log transformed to achieve normal distribution and stabilize variance. For example, for a Ki67 score of 30%, the recorded transformed value would be log2(30.1). The geometric mean of Ki67 (by taking the antilogs of the means on log2 scale) across the 100 cases ranged from 7.1% (Laboratory G) to 23.9% (Laboratories D and F) (the arithmetic mean ranged from 15.6% to 31.1%). Such a range indicates substantial differences in Ki67 measurement across laboratories on centrally stained slides from the same cases.

![boxplots](http://jnci.oxfordjournals.org/content/105/24/1897/F2.large.jpg)

Side-by-side boxplots of Ki67 distribution from Experiments 2A and 2B. A) Centrally stained, local scoring method. B) Locally stained, local scoring method.

+  _Local Staining, Local Scoring Method_  
Table 3 provides, for each laboratory, the summary statistics of log2-transformed Ki67 scores that were observed when all laboratories used their own local scoring methods on sections they stained according to their own local staining methods. The geometric mean of Ki67 across the 100 cases ranged from 6.1% (Laboratory G) to 30.1% (Laboratory F) (the arithmetic mean ranged from 12.6% to 35.5%), suggesting a large interlaboratory variability in Ki67 measurement.

![dot plot](http://jnci.oxfordjournals.org/content/105/24/1897/F3.large.jpg)

Dot plot of Ki67 measurements by laboratory from Experiments 2A and 2B. A) Centrally stained, local scoring method. B) Locally stained, local scoring method.

__Degree of concordance__

![degree of concordance](http://jnci.oxfordjournals.org/content/105/24/1897/F5.large.jpg)

Degree of concordance (bolded box) between Laboratory B and Laboratory D at the St. Gallen–recommended Ki67 percentage of positively stained tumor cells cutoff of 13.5%. At a hypothetical 13.5% cutoff, there are 32.3% cases that Laboratory D would call high Ki67 but Laboratory B would call low Ki67.

---


[**An international study to increase concordance in Ki67 scoring**](http://www.nature.com/modpathol/journal/v28/n6/full/modpathol201538a.html)  


Read it later but some graphs may be of use

![log2 transformed summary statistics](https://raw.githubusercontent.com/tt6746690/CSC148/master/299manuscript/img/summary%20statistics.PNG)   

Summary statistics of log2-transformed Ki67 scores (percentage of invasive cancer cells scored positive), scoring on glass tissue microarray slides (16 laboratories, 50 cases)

![variability in ki67 scoring](https://raw.githubusercontent.com/tt6746690/CSC148/master/299manuscript/img/ki67%20variability.PNG)

Figure 3 displays the variation in scores across laboratories for the 50 cases. Highlighted (gray lines) are the 26 cases for which at least one of the 16 laboratories reported a score in the clinically relevant range of 10%≤Ki67≤20%. Within each Group, laboratories were looking at the exact same tissue microarray section; between Groups, the laboratories were looking at different sections derived from the same tissue microarray block.



---
[_A novel model for Ki67 assessment in breast cancer_](https://diagnosticpathology.biomedcentral.com/articles/10.1186/1746-1596-9-118)

+ has some quite useful info  
  + 20% cutoff is taken  


[_A methodology to ensure and improve accuracy
of Ki67 labelling index estimation by automated
digital image analysis in breast cancer tissue_](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4053156/)

+ used Aperio software
+ normally score 1000 cells but 500 is bare minimum
+ traditional methods
  + scanning the image quickly and guess the percentage
  + identify hotspot for ki67, and then count  
+ the statistics is quite useful


[_Digital immunohistochemistry wizard: image analysis-assisted stereology tool to produce reference data set for calibration and quality control_](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4305978/figure/F4/)  

+ have some interesting graphics, could borrow from that
