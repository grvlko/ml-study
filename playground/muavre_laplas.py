import pandas as pd
import scipy.stats as stats
from math import sqrt, exp, pi, ceil

def standard_normal_distr_density(x):
    return exp(-x**2 / 2) / sqrt(2 * pi)

def compute_probabilities(n, p):
    q, mid, mu = 1 - p, n / 2, n * p
    sigma, k_star = sqrt(n * p * q), round((n + 1) * p)
    lower, upper = max(0, ceil(mid - sigma)), min(n, round(mid + sigma))

    # Точные значения (биномиальное распределение)
    exact_interval = stats.binom.cdf(upper, n, p) - stats.binom.cdf(lower - 1, n, p)
    exact_cdf = stats.binom.cdf(5, n, p)
    exact_pmf = stats.binom.pmf(k_star, n, p)

    # Приближение Пуассона
    poisson_lambda = n * p
    poisson_interval = stats.poisson.cdf(upper, poisson_lambda) - stats.poisson.cdf(lower - 1, poisson_lambda)
    poisson_cdf = stats.poisson.cdf(5, poisson_lambda)
    poisson_pmf = stats.poisson.pmf(k_star, poisson_lambda)

    # Приближение Пуассона
    local_interval = sum(standard_normal_distr_density(k) / sigma for k in range(lower, upper + 1))
    local_cdf = sum(standard_normal_distr_density(k) / sigma for k in range(6))
    local_pmf = standard_normal_distr_density(k_star) / sigma

    # Интегральная теорема Муавра-Лапласа
    integral_interval = stats.norm.cdf((upper - mu) / sigma) - stats.norm.cdf((lower - 1 - mu) / sigma)
    integral_cdf = stats.norm.cdf((5 - mu) / sigma)
    integral_pmf = stats.norm.cdf((k_star - mu) / sigma) - stats.norm.cdf((k_star - 1 - mu) / sigma)

    return {
        "exact": (exact_interval, exact_cdf, exact_pmf),
        "poisson": (poisson_interval, poisson_cdf, poisson_pmf),
        "local_ml": (local_interval, local_cdf, local_pmf),
        "integral_ml": (integral_interval, integral_cdf, integral_pmf),
    }

# Параметры
n_values = [100, 1000, 10000]
p_values = [0.001, 0.01, 0.1, 0.25, 0.5]

data = []
index = []
for n in n_values:
    for p in p_values:
        index_label = f"n={n}, p={p}"
        index.append(index_label)
        results = compute_probabilities(n, p)
        data.append({
            'exact P([...])': results["exact"][0],
            'exact P(S <= 5)': results["exact"][1],
            'exact P(S = k*)': results["exact"][2],
            
            'poisson P([...])': results["poisson"][0],
            'poisson P(S <= 5)': results["poisson"][1],
            'poisson P(S = k*)': results["poisson"][2],
            
            'local_ml P([...])': results["local_ml"][0],
            'local_ml P(S <= 5)': results["local_ml"][1],
            'local_ml P(S = k*)': results["local_ml"][2],
            
            'integral_ml P([...])': results["integral_ml"][0],
            'integral_ml P(S <= 5)': results["integral_ml"][1],
            'integral_ml P(S = k*)': results["integral_ml"][2],
        })

df = pd.DataFrame(
    data, 
    index=index,
    columns=[
        'exact P([...])', 'exact P(S <= 5)', 'exact P(S = k*)',
        'poisson P([...])', 'poisson P(S <= 5)', 'poisson P(S = k*)',
        'local_ml P([...])', 'local_ml P(S <= 5)', 'local_ml P(S = k*)',
        'integral_ml P([...])', 'integral_ml P(S <= 5)', 'integral_ml P(S = k*)'
    ]
)

df.to_csv('bernoulli_results.csv', float_format='%.3f', index=True)
print(df)