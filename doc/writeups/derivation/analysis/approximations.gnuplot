mua = 1200
k = 1
rho = 1
c = 1
alpha = k / rho / c
E0 = 1
z0 = 0
d = 12e-4
z = 0.0006

A(tp) = mua*sqrt(alpha*tp)
B(tp) = (z0-z) / (4*alpha*tp)**0.5
C(tp) = d / (4*alpha*tp)**0.5

set samples 1000
set xrange[0:1e-4]
set yrange[0:]
exact(t) = (mua*E0/rho/c/2)*(exp(-mua*(z-z0)))*exp(A(t)**2) * ( erf(A(t)+B(t)+C(t)) - erf(A(t)+B(t)) )
approx2(t) = (mua*E0/rho/c/2)*(exp(-mua*(z-z0)))*( exp(-B(t)**2 - 2*A(t)*B(t))/sqrt(pi)/(A(t) + B(t))  - exp(-B(t)**2 - C(t)**2 - 2*A(t)*B(t) - 2*A(t)*C(t) - 2*B(t)*C(t))/sqrt(pi)/(A(t) + B(t) + C(t)) )
print exact(0.0000000001)
plot exact(x), approx2(x), A(x)+B(x)+C(x) axis x1y2, A(x) + B(x) axis x1y2



