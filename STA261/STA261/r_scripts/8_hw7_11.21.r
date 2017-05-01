x <- c(3.03, 5.53, 5.6, 9.3, 9.92, 12.51, 12.95, 15.21, 16.04, 16.84)
y <- c(3.19, 4.26, 4.47, 4.53, 4.67, 4.68, 12.78, 6.79, 9.37, 12.75)

pool_var <- ( (10 - 1)*var(x) + (10 -1)*var(y) ) / (10 + 10 -2)
t <- (mean(x) - mean(y)) / (pool_var * sqrt(1 / 10 + 1 / 10))
f <- var(x) / var(y)

p_value <- 2*( 1-pt(t, df=18) )

print(pool_var)
print(t)
print(p_value)

print(f)




