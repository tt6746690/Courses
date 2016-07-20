

_box cox test to determine transformation factor_

```r
# check qq plot
qqnorm(x)
qqline(x)
# box cox
library('MASS')
b <- boxcox(Volume~Height+girth, data=trees)

```
