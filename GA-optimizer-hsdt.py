import numpy as np
import random
import warnings
from deap import base, creator, tools, algorithms
from scipy.integrate import dblquad
from scipy.linalg import eig
from scipy.stats import qmc

warnings.filterwarnings('ignore')

# ============= Material Properties =============
E_c = 348.43e9
E_m = 201.04e9
rho_c = 2370
rho_m = 8166
nu = 0.3

# ============= Beam Dimensions =============
L = 1.0
h = 0.02
MAX_WEIGHT = 50
MIN_THICKNESS = 0.003

# ============= Tight Optimization Constraints =============
N_MIN, N_MAX = 0.1, 5
K_MIN, K_MAX = 0.1, 5
ALPHA_MIN, ALPHA_MAX = 0.01, 0.4
BH_MIN, BH_MAX = 0.10, 0.8

# ============= GA Parameters =============
GA_POP_SIZE = 50
GA_NGEN = 30
GA_CXPB = 0.7
GA_MUTPB = 0.8
GA_TOURNSIZE = 5
GA_MUT_SIGMA = 0.01
GA_MUT_INDPB = 0.5

def clamp(x, a, b):
    try:
        return max(min(float(x), b), a)
    except Exception:
        return a  # fallback

def repair_individual(individual):
    try:
        individual[0] = float(clamp(individual[0], N_MIN, N_MAX))
        individual[1] = float(clamp(individual[1], K_MIN, K_MAX))
        individual[2] = float(clamp(individual[2], ALPHA_MIN, ALPHA_MAX))
        individual[3] = float(clamp(individual[3], BH_MIN, BH_MAX))
    except Exception as e:
        individual[:] = [1.0, 1.0, 0.05, 0.3]
    return individual

class FGBeamHSDT:
    def __init__(self, L, h, E_c, E_m, rho_c, rho_m, nu):
        self.L = L
        self.h = h
        self.E_c = E_c
        self.E_m = E_m
        self.rho_c = rho_c
        self.rho_m = rho_m
        self.nu = nu
        self.MIN_E = 1e6
        self.MIN_RHO = 1
        self.EPS = 1e-12

    def effective_properties(self, y, z, b, h, n, k, alpha, porosity_type):
        z_bar = (z + h/2) / h
        V_c = 1 - z_bar ** n
        if porosity_type == 'uniform':
            f_p = 1.0
        elif porosity_type == 'non-uniform':
            y_abs = min(abs(y), b/2 - 1e-12)
            z_abs = min(abs(z), h/2 - 1e-12)
            f_p = (1 - 2*y_abs/b) * (1 - 2*z_abs/h)
        else:
            f_p = 1.0
        E_eff = ((self.E_c - self.E_m)*V_c + self.E_m - (self.E_c - self.E_m)*(alpha/2)*f_p)
        rho_eff = ((self.rho_c - self.rho_m)*V_c + self.rho_m - (self.rho_c - self.rho_m)*(alpha/2)*f_p)
        return max(self.MIN_E, E_eff), max(self.MIN_RHO, rho_eff)

    def shape_functions(self, z, h):
        z_safe = max(min(z, h/2 - 1e-12), -h/2 + 1e-12)
        f_z = z_safe - (h/np.pi)*np.sin(np.pi*z_safe/h)
        g_z = np.cos(np.pi*z_safe/h)
        return f_z, g_z

    def integrate_property(self, func, b, h, n, k, alpha, porosity_type, epsabs=1e-6, epsrel=1e-6):
        def integrand(z, y):
            try:
                E, rho = self.effective_properties(y, z, b, h, n, k, alpha, porosity_type)
                return func(z, y, E, rho)
            except:
                return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result, _ = dblquad(
                    integrand,
                    -b/2 + 1e-12, b/2 - 1e-12,
                    lambda y: -h/2 + 1e-12,
                    lambda y: h/2 - 1e-12,
                    epsabs=epsabs,
                    epsrel=epsrel
                )
                return result if not np.isnan(result) else 0.0
            except:
                return 0.0

    def calculate_stiffness_terms(self, b, h, n, k, alpha, porosity_type):
        nu = self.nu
        denom = max(1e-12, 1 - nu**2)
        integrands = {
            'A11': lambda z, y, E, rho: E/denom,
            'B11': lambda z, y, E, rho: z*E/denom,
            'D11': lambda z, y, E, rho: z**2*E/denom,
            'B11s': lambda z, y, E, rho: self.shape_functions(z, h)[0]*E/denom,
            'D11s': lambda z, y, E, rho: z*self.shape_functions(z, h)[0]*E/denom,
            'H11s': lambda z, y, E, rho: self.shape_functions(z, h)[0]**2*E/denom,
            'A55s': lambda z, y, E, rho: self.shape_functions(z, h)[1]**2*E/(2*(1 + nu))
        }
        results = {name: self.integrate_property(func, b, h, n, k, alpha, porosity_type)
                   for name, func in integrands.items()}
        return results

    def calculate_inertia_terms(self, b, h, n, k, alpha, porosity_type):
        f_z = lambda z: self.shape_functions(z, h)[0]
        integrands = {
            'J0': lambda z, y, E, rho: rho,
            'J1': lambda z, y, E, rho: z*rho,
            'J2': lambda z, y, E, rho: f_z(z)*rho,
            'J3': lambda z, y, E, rho: z**2*rho,
            'J4': lambda z, y, E, rho: z*f_z(z)*rho,
            'J5': lambda z, y, E, rho: f_z(z)**2*rho
        }
        results = {name: self.integrate_property(func, b, h, n, k, alpha, porosity_type)
                   for name, func in integrands.items()}
        return results

    def calculate_frequency(self, b, h, n, k, alpha, porosity_type, mode=0):
        beta = np.pi / self.L
        S = self.calculate_stiffness_terms(b, h, n, k, alpha, porosity_type)
        I = self.calculate_inertia_terms(b, h, n, k, alpha, porosity_type)
        try:
            K = np.array([
                [S['A11']*beta**4, -S['B11']*beta**3, -S['B11s']*beta**3],
                [-S['B11']*beta**3, S['D11']*beta**2 + S['A55s']*beta**4, S['D11s']*beta**2 + S['A55s']*beta**4],
                [-S['B11s']*beta**3, S['D11s']*beta**2 + S['A55s']*beta**4, S['H11s']*beta**2 + S['A55s']*beta**4]
            ])
            M = np.array([
                [I['J0'], -I['J1'], -I['J2']],
                [-I['J1'], I['J3'], I['J4']],
                [-I['J2'], I['J4'], I['J5']]
            ])
            eigvals, _ = eig(K, M)
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) == 0:
                return 1e-3
            freq = np.sqrt(eigvals[mode]) / (2 * np.pi)
            return freq
        except Exception:
            return 1e-3

    def calculate_weight(self, b, h, n, k, alpha, porosity_type):
        samples = 6
        weight = 0
        for i in range(samples):
            y = b * (i/(samples-1) - 0.5)
            for j in range(samples):
                z = h * (j/(samples-1) - 0.5)
                _, rho = self.effective_properties(y, z, b, h, n, k, alpha, porosity_type)
                weight += rho
        return weight * (b*h)/samples**2 * L

# DEAP setup: Single objective (efficiency)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

class FGBeamOptimizerGA:
    def __init__(self, L, h, E_c, E_m, rho_c, rho_m, nu):
        self.L = L
        self.h = h
        self.E_c = E_c
        self.E_m = E_m
        self.rho_c = rho_c
        self.rho_m = rho_m
        self.nu = nu
        self.beam = FGBeamHSDT(L, h, E_c, E_m, rho_c, rho_m, nu)
        self.fail_count = 0

    def evaluate(self, individual, porosity_type):
        if not isinstance(porosity_type, str) or porosity_type not in ['uniform', 'non-uniform']:
            porosity_type = 'uniform'
        if not (isinstance(individual, list) and len(individual) == 4 and
                all(isinstance(x, (int, float, np.floating, np.integer, str)) for x in individual)):
            individual = [1.0, 1.0, 0.05, 0.3]
        try:
            individual = [float(x) for x in individual]
        except Exception:
            individual = [1.0, 1.0, 0.05, 0.3]
        repair_individual(individual)
        n, k, alpha, b_h = individual
        b = max(b_h * self.h, MIN_THICKNESS)
        try:
            freq = self.beam.calculate_frequency(b, self.h, n, k, alpha, porosity_type)
            weight = self.beam.calculate_weight(b, self.h, n, k, alpha, porosity_type)
        except Exception as e:
            freq, weight = 1e-3, MAX_WEIGHT*2
        if np.isnan(freq) or np.isnan(weight) or weight > MAX_WEIGHT*10 or b < MIN_THICKNESS:
            self.fail_count += 1
            return 1e-8,
        if weight > MAX_WEIGHT:
            freq *= (MAX_WEIGHT/weight)
        if freq < 1e-8:
            freq = 1e-8
        fitness = freq / weight
        return fitness,

def optimize_beam(porosity_type):
    optimizer = FGBeamOptimizerGA(L, h, E_c, E_m, rho_c, rho_m, nu)
    toolbox = base.Toolbox()
    toolbox.register("attr_n", random.uniform, N_MIN, N_MAX)
    toolbox.register("attr_k", random.uniform, K_MIN, K_MAX)
    toolbox.register("attr_alpha", random.uniform, ALPHA_MIN, ALPHA_MAX)
    toolbox.register("attr_bh", random.uniform, BH_MIN, BH_MAX)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                   (toolbox.attr_n, toolbox.attr_k, toolbox.attr_alpha, toolbox.attr_bh), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=GA_MUT_SIGMA, indpb=GA_MUT_INDPB)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNSIZE)
    toolbox.register("evaluate", optimizer.evaluate, porosity_type)

    pop = toolbox.population(n=GA_POP_SIZE)
    # عينات خاصة
    for i in range(5):
        pop[i] = creator.Individual([1.0, 1.0, 0.05, 0.3])
    for i in range(5, 15):
        pop[i] = creator.Individual([
            random.uniform(N_MIN, N_MAX),
            random.uniform(K_MIN, K_MAX),
            random.uniform(ALPHA_MIN, ALPHA_MAX),
            random.uniform(BH_MIN, BH_MAX)
        ])
    pop[15] = creator.Individual([N_MIN, K_MIN, ALPHA_MIN, BH_MIN])
    pop[16] = creator.Individual([N_MAX, K_MAX, ALPHA_MAX, BH_MAX])
    pop[17] = creator.Individual([N_MIN, K_MAX, ALPHA_MIN, BH_MAX])
    pop[18] = creator.Individual([N_MAX, K_MIN, ALPHA_MAX, BH_MIN])
    sampler = qmc.LatinHypercube(d=4)
    samples = sampler.random(n=10)
    for i in range(10):
        pop[20+i] = creator.Individual([
            N_MIN + samples[i,0]*(N_MAX-N_MIN),
            K_MIN + samples[i,1]*(K_MAX-K_MIN),
            ALPHA_MIN + samples[i,2]*(ALPHA_MAX-ALPHA_MIN),
            BH_MIN + samples[i,3]*(BH_MAX-BH_MIN)
        ])
    # إصلاح وضبط السكان الأولي
    new_pop = []
    for ind in pop:
        if isinstance(ind, list) and len(ind) == 4 and all(isinstance(x, (int, float, np.floating, np.integer)) for x in ind):
            repair_individual(ind)
            new_pop.append(ind)
    pop = new_pop

    # تقييم أولي
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    for gen in range(GA_NGEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB)
        valid_offspring = []
        for ind in offspring:
            if isinstance(ind, list) and len(ind) == 4 and all(isinstance(x, (int, float, np.floating, np.integer)) for x in ind):
                repair_individual(ind)
                valid_offspring.append(ind)
        fits = list(map(toolbox.evaluate, valid_offspring))
        for ind, fit in zip(valid_offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(valid_offspring + pop, GA_POP_SIZE)
        hof.update(pop)
    return optimizer.beam, hof[0]

def show_results(beam, best_params, porosity_type):
    if not best_params or len(best_params) != 4:
        print("No valid solution found!")
        return
    n, k, alpha, b_h = best_params
    b = b_h * h
    freq = beam.calculate_frequency(b, h, n, k, alpha, porosity_type)
    weight = beam.calculate_weight(b, h, n, k, alpha, porosity_type)
    eff = freq / weight
    thick_check = "OK" if b >= MIN_THICKNESS else "VIOLATED"
    weight_check = "OK" if weight <= MAX_WEIGHT else "VIOLATED"
    constraint = "OK" if (thick_check == "OK" and weight_check == "OK") else "VIOLATED"
    print(f"{porosity_type.capitalize():<15} | "
          f"{n:7.3f} | {k:7.3f} | {alpha:7.3f} | {b_h:7.3f} | {b:7.4f} | "
          f"{freq:12.3f} | {weight:10.3f} | {eff:14.3f} | "
          f"{thick_check:>8} | {weight_check:>8} | {constraint:>9}")

if __name__ == "__main__":
    print("\nOptimization Results Summary (Table)\n")
    print(f"{'Porosity':<15} | {'n':>7} | {'k':>7} | {'alpha':>7} | {'b/h':>7} | {'b (m)':>7} | "
          f"{'Freq (Hz)':>12} | {'Weight (kg)':>10} | {'Efficiency':>14} | {'Min Thk':>8} | {'Max Wt':>8} | {'Constraint':>9}")
    print("-"*139)
    print(f"{'Range':<15} | "
          f"{f'[{N_MIN:.2f},{N_MAX:.2f}]':>7} | "
          f"{f'[{K_MIN:.2f},{K_MAX:.2f}]':>7} | "
          f"{f'[{ALPHA_MIN:.2f},{ALPHA_MAX:.2f}]':>7} | "
          f"{f'[{BH_MIN:.2f},{BH_MAX:.2f}]':>7} | "
          f"{'':>7} | {'':>12} | {'':>10} | {'':>14} | {'':>8} | {'':>8} | {'':>9}")
    print("-"*139)
    beam, best_params = optimize_beam('uniform')
    show_results(beam, best_params, 'uniform')

    beam, best_params = optimize_beam('non-uniform')
    show_results(beam, best_params, 'non-uniform')
    print("-"*139)
