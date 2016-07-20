

__Supervised learning__
+ Supervised learning is the machine learning task of inferring a function from labeled training data.

_Regression problem_  
+ predict continous valued outputs

_Classification problem_  
+ Predict a discrete valued output (maybe > 2).    
+ For example, predict degree of malignancy from tumour size, patient age (2 features). The role of a learning algorithm aims to separate the outcomes with a straight line.   
+ SVM (support vector machine) helps to deal with infinite number of features,

> It's important to determine which kind of a problem depending on if the output value is continuous or discrete


__Unsupervised learning__
+ The machine learning task of inferring a function to describe hidden structure from unlabeled data. Since the examples given to the learner are unlabeled, there is no error or reward signal to evaluate a potential solution.   



_Usual procedures_

+ Training set   
+ Learning algorithm  
+ Hypothesis function  

> when the input is x and output is y. then y = f(x) is the target function. However f(x) is unknown and the goal of the learning algorithm is to find the best hypothesis h(x), which approximates f(x)


![](https://qph.is.quoracdn.net/main-qimg-a7bec039b4badd40c5ed4051a0f56d09?convert_to_webp=true)



_Cost function_

Set up function `h(x) = θ0 + θ1x` so that `h(x)` is close to y for training set (x, y). An obvious example would be selecting `θ0` and `θ1` so as to minimize `(1/2m)sum((h(x) - y)^2)` where m is the number of instances in the training set.

![cost function](https://raw.githubusercontent.com/tt6746690/courseProjects/master/iGEM2016/images/cost%20function.png)


_examples of cost function_


![cost function summary](https://raw.githubusercontent.com/tt6746690/courseProjects/master/iGEM2016/images/cost%20function%20summary.png)

![next slide](https://raw.githubusercontent.com/tt6746690/courseProjects/master/iGEM2016/images/s.png)
