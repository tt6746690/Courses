
alpha <- runif(1)	# random init value 
tolerance <- 5e-16 	# largest difference between successive alphas 
delta <- 1
LogLikeOld <- sum(log(1+alpha*x))	# log-likelihood at initial alpha 

while(delta > tolerance){
	lPrime <- sum(x/(1+alpha*x))	# first derivative 
	l2Prime <- -sum(x^2/(1+alpha*x)^2)	# second derivative 
	alpha <- alpha - lPrime / l2Prime	#newton-raphson update 
	LogLike <- sum(log(1+alpha*x))		# log likelihood at new alpha 
	delta <- abs(LogLike - LogLikeOld_	# Difference 
	LogLikeOld <- LogLike 
}
