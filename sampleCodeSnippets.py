

# Draw samples from uniform distribution
from scipy.stats import uniform
scale = hk_max - hk_min
hk_sample = uniform.rvs(loc=hk_min, scale=scale, size=size)