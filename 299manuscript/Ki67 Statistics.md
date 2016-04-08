
[__The Bland-Altman Plot__](https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot)  
A Bland–Altman plot (Difference plot) in analytical chemistry and biostatistics is a method of data plotting used in analyzing the agreement between two different assays.

![example blant-altman plot](https://upload.wikimedia.org/wikipedia/commons/f/f8/Bland-Alman_Plot_with_CI%27s_on_LOA.png)

_Agreement vs. Correlation_
Bland and Altman make the point that any two methods that are designed to measure the same parameter (or property) should have good correlation when a set of samples are chosen such that the property to be determined varies considerably. A high correlation for any two methods designed to measure the same property could thus in itself just be a sign that one has chosen a widespread sample. A high correlation does not automatically imply that there is good agreement between the two methods.

_Constructing the plot_
the Cartesian coordinates of a given sample S with values of S_1 and S_2 determined by the two assays is

`S(x,y) = ((S1+S2) / 2, S1 - S2)`

For comparing the dissimilarities between the two sets of samples independently from their mean values, it is more appropriate to look at the __ratio__ of the pairs of measurements. One primary application of the Bland–Altman plot is to compare two clinical measurements that each provide some errors in their measure. It can also be used to compare a new measurement technique or method with a gold standard, as even a gold standard does not - or should not - imply it to be without error. Bland and Altman plots allow us to investigate the existence of any systematic difference between the measurements (i.e., fixed bias) and to identify possible outliers. The mean difference is the estimated bias, and the SD of the differences measures the random fluctuations around this mean.

_Limit of agreement and confidence levels_
It is common to compute 95% limits of agreement for each comparison (average difference ± 1.96 standard deviation of the difference), which tell us how far apart measurements by 2 methods were more likely to be for most individuals. If the differences within mean ± 1.96 SD are not clinically important, the two methods may be used interchangeably. The 95% limits of agreement can be unreliable estimates of the population parameters especially for small sample sizes so, when comparing methods or assessing repeatability, it is important to calculate __confidence intervals__ for 95% limits of agreement. This can be done by Bland and Altman's approximate method or by more precise methods

---

_PAPER: _[__Statistical methods for assessing agreement between two methods of clinical measurements__](https://www-users.york.ac.uk/~mb55/meas/ba.pdf)  

Instead indirect methods are used, and a new method has to be evaluated by comparison with an established technique rather than with the true quantity. If the new method agrees sufficiently well with the old, the old may be replaced. This is very different from calibration, where known quantities are measured by a new method and the result compared with the true value or with measurements made by a highly accurate method. When two methods are compared neither provides an unequivocally correct measurement, so we try to assess the degree of agreement.

> Often, product-moment correlation coefficient between the results of the two measurements where used as indicator of agreement. There is no such thing!

We want to know by how much the new method is likely to differ from the old: if this is not enough to cause problems in clinical interpretation we can replace the old method by the new or use the two interchangeably.

_STEPS_

_Ploting the data_
1. The first step is to plot the data and draw the __line of equality__ on which all points would lie if
the two meters gave exactly the same reading every time. This helps the eye in gauging
the degree of agreement between measurements  

![linear correlation](https://raw.githubusercontent.com/tt6746690/CSC148/master/299manuscript/img/linear%20correlation%20bland-altman%20example.PNG)


2. The second step is usually to calculate the __correlation coefficient__ (r) between the two
methods. The null hypothesis here is that the measurements by the two methods are not linearly related. If the probability is very small then we can safely conclude that PEFR measurements by the mini and large meters are related.
_However_, this high correlation does not mean that the two methods agree.  
    + r measures the strength of a relation between two variables, not the agreement between them. We have perfect agreement only if the points in fig 1 lie along the line of equality, but we will have perfect correlation if the points lie along any straight line.
    + A change in scale of measurement does not affect the correlation, but it certainly affects the agreement. This can be certainly a problem in determining KI67 positive percentage. A computer aided scoring that consistently overestimate positivity may not change correlation but indeed affect agreement.  
    +  Since investigators usually try to compare two methods over the whole range of values typically encountered, a high correlation is almost guaranteed ( Central Limit Theorem )  
    + The test of significance may show that the two methods are related, but it would be amazing if two methods designed to measure the same quantity were not related. The test of significance is irrelevant to the question of agreement.
    +  Data which seem to be in poor agreement can produce quite high correlations.  

_Measuring agreement_
How far apart measurements can be without causing difficulties will be a question of judgment. Ideally, it should be defined in advance to help in the interpretation of the method comparison and to choose the sample size.

A plot of the difference between the methods against their mean may be more informative than a simple plot of one method against the other, without a regression line.  The __plot of difference against mean__ also allows us to investigate any possible relationship between the measurement error and the true value.

![altman bland plot](https://raw.githubusercontent.com/tt6746690/CSC148/master/299manuscript/img/difference%20against%20mean%20for%20PEPR%20data.PNG)


For the PEFR data, there is no obvious relation between the difference and the mean. Under these circumstances we can summarise the lack of agreement by calculating the __bias__, estimated by the mean difference d and the standard deviation of the differences (s ).  If there is a consistent bias we can adjust for it by subtracting d from the new method. We would expect most of the differences to lie between d - 2s and d + 2s. Such differences are
likely to follow a __Normal distribution__ because we have removed a lot of the variation between subjects and are left with the measurement error.

The measurements themselves do not have to follow a Normal distribution, and often they will not. We can check the distribution of the differences by drawing a __histogram__. If this is skewed or has very long tails the assumption of
Normality may not be valid

_limits of agreement_  

```python
d - 2s = -2.1- (2 X 38.8) = -79.7 l/min
d + 2s = -2.1 + (2 X 38.8) = -75.5 l/min
```

Thus, the mini meter may be 80 l/min below or 76 l/min above the large meter, which would be unacceptable for clinical purposes. This lack of agreement is by no means obvious in fig 1.


_Precision of estimated limits of agreement_
We might sometimes wish to use standard errors and confidence intervals to see how precise our estimates are, provided the differences follow a distribution which is approximately Normal. The __standard error__ of `d` is `sqrt(s^2 / n)`
, where n is the sample size, and the standard error of `d - 2s` and `d + 2s` is about `cuberoot(3s^2 / n)`. 95% __confidence intervals__ can be calculated by finding the appropriate point of the t distribution with n -1 degrees of freedom, on most tables the columns marked 5% or 0.05, and then the confidence interval will be from the observed value minus t standard errors to the observed value plus t standard errors.



---

#### Inter-rater reliability

In statistics, inter-rater reliability, inter-rater agreement, or concordance is the degree of agreement among raters. It gives a score of how much homogeneity, or consensus, there is in the ratings given by judges.

_Soure of disagreement_
+ For any task in which multiple raters are useful, raters are expected to disagree about the observed target.  
+ By contrast, situations involving unambiguous measurement, such as simple __counting__ tasks (e.g. number of potential customers entering a store), often do not require more than one person performing the measurement.   

Measurement involving ambiguity in characteristics of interest in the rating target are generally improved with multiple trained raters. Such measurement tasks often involve subjective judgment of quality (examples include ratings of physician 'bedside manner', evaluation of witness credibility by a jury, and presentation skill of a speaker). Without scoring guidelines, ratings are increasingly affected by experimenter's bias, that is, a tendency of rating values to drift towards what is expected by the rater. During processes involving repeated measurements, correction of rater drift can be addressed through periodic retraining to ensure that raters understand guidelines and measurement goals.


_Pearson's r_  
can be used to measure __pairwise__ correlation among raters using a scale that is ordered. It assumes that the rating scale is continuous. It's important to note that it only considers relative positions.  

_Intraclass correlation coefficient_
There are several types of this and one is defined as,

> "the proportion of variance of an observation due to between-subject variability in the true scores".

 The range of the ICC may be between 0.0 and 1.0. The ICC will be high when there is little variation between the scores given to each item by the raters, e.g. if all raters give the same, or similar scores to each of the items. The ICC is an improvement over Pearson's r, as it takes into account the differences in ratings for individual segments, along with the correlation between raters.

 _Limit of agreement_
 Another approach to agreement (useful when there are only __two raters__ and the scale is continuous) is to calculate the differences between each pair of the two raters' observations. The mean of these differences is termed __bias__ and the reference interval (mean +/- 1.96 x standard deviation) is termed __limits of agreement__. The limits of agreement provide insight into how much random variation may be influencing the ratings. __Confidence limits__ (usually 95%) can be calculated for both the bias and each of the limits of agreement.  

The __Bland–Altman plot__ demonstrates not only the overall degree of agreement, but also whether the agreement is related to the underlying value of the item. For instance, two raters might agree closely in estimating the size of small items, but disagree about larger items.
