# Alex Caselli      Assignment 5

import numpy as np
from math import cos, pi
import matplotlib.pyplot as plt
import random
from random import randint
from tqdm import tqdm
import os

# Directory to store data
directory = '/Desktop/plots'
if not os.path.exists(directory):
  os.makedirs(directory)

# x_i in [-5, 5]


# Functions
def SphereFunction(x):
    if (len(x.shape) == 3):
        return np.sum(np.square(x), axis=0)
    else:
        return np.sum(np.square(x), axis=1)

def rastriginFunction(x, A=10):
    if (len(x.shape) == 3):
        n = x.shape[0]
        summy = np.zeros(x.shape[1:2])
        for i in range(n):
            summy = summy + ( x[i,:,:]**2 - A * np.cos(2*pi*x[i,:,:]) )
    else:
        n = x.shape[1]
        summy = np.zeros(x.shape[0])
        for i in range(n):
            summy = summy + ( x[:,i]**2 - A * np.cos(2*pi*x[:,i]) )

    c = A * n
    
    return c + summy



# Plot the functions
def visualizeSphere(x,y, min_val, max_val):
    X, Y = np.meshgrid(x, y)
    Z = SphereFunction(np.stack([X, Y], axis=0))
    plt.figure()
    contours = plt.contour(X, Y, Z, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.imshow(Z, extent=[min_val, max_val, min_val, max_val], origin='lower',
           cmap='RdGy', alpha=0.5)
    plt.colorbar(); 
    
def visualizeRastrigin(x,y, min_val, max_val):

    X, Y = np.meshgrid(x, y)
    Z = rastriginFunction(np.stack([X, Y], axis=0))
    plt.figure()
    contours = plt.contour(X, Y, Z, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.imshow(Z, extent=[min_val, max_val, min_val, max_val], origin='lower',
           cmap='RdGy', alpha=0.5)
    plt.colorbar(); 


   

# generate samples to plot on the contour plots
def generateSamples(min_val, max_val):
    x = np.linspace(min_val, max_val, 100)
    y = np.linspace(min_val, max_val, 100)


    # generate random points
    np.random.seed(0)
    g = np.random.default_rng().uniform(min_val+0.5, max_val-0.5, (100, 2))

    #Plot points on Sphere 
    visualizeSphere(x,y, min_val, max_val)
  
    plt.scatter(g[:,0], g[:,1], c=SphereFunction(g), s=20, cmap='Spectral_r')
    plt.show()

    #Plot points on Rast
    visualizeRastrigin(x,y, min_val, max_val)

    plt.scatter(g[:,0], g[:,1], c=rastriginFunction(g), s=20, cmap='Spectral_r')
    plt.show()





# Plot generations
def plot_visualize_sphere(n_samples, generation, min_val, max_val):
    x = np.linspace(min_val, max_val, n_samples)
    y = np.linspace(min_val, max_val, n_samples)
    visualizeSphere(x,y, min_val, max_val)
    plt.scatter(generation[:,0], generation[:,1], c=SphereFunction(generation), s=20, cmap='Spectral_r')
    


def plot_visualize_rastrigin(n_samples, generation, min_val, max_val):
    x = np.linspace(min_val , max_val, 1000)
    y = np.linspace(min_val, max_val, 1000)
    visualizeRastrigin(x,y, min_val , max_val)
    plt.scatter(generation[:,0], generation[:,1], c=rastriginFunction(generation), s=20, cmap='Spectral_r')


# CEM
class CEM:

    def __init__(self):
        super().__init__()

        # Initialize mean and standard deviation
        #theta_mean = np.zeros((n_samples, dim_theta))
        theta_mean = np.random.uniform(min_val, max_val, (n_samples, dim_theta))
        theta_std = np.random.uniform(max_val-1, max_val, (n_samples, dim_theta))
        self.n_samples = n_samples
        self.t = n_iterations
        self.top_p = top_p
        self.fit_gaussian(theta_mean, theta_std)

    def fit_gaussian(self, mean, std):
        # theta is actualy the population sampled from the distribution
        theta = np.random.normal(mean, std)
        self.theta = np.clip(theta, min_val, max_val)

        

    def generation(self, function=0):
        # Sample n_sample candidates from N(theta)
        mean_fitness = []
        best_fitness = []
        worst_fitness = []
        for i in tqdm(range(0, self.t)):
            #0 --> Sphere; 1 --> Rasti
            fitness = self.evaluate_fitness(self.theta, function)

            mean_fitness.append(np.mean(fitness))
            best_fitness.append(np.min(fitness))
            worst_fitness .append(np.max(fitness))

            couple = list(zip(self.theta, fitness.T))
            sorted_fitness= sorted(couple, key=lambda tup: tup[1])
            elite = self.take_elite(sorted_fitness)
           

            e_candidates = [i[0] for i in elite]
            if plot == 1:
                self.plot_candidates(self.theta, function, min_val, max_val)
            plt.pause(pause)
            new_mean = self.compute_new_mean(e_candidates)
            new_std = self.compute_new_std(e_candidates)
            self.fit_gaussian(new_mean, new_std)
            if plot == 1:
                plt.close("all")
        if plot == 1:        
            plt.show()

        plt.plot(mean_fitness)
        plt.show()
        return mean_fitness, best_fitness, worst_fitness
        
        


    def take_elite(self, candidates):
        n_top = int((self.n_samples * self.top_p)/ 100)
        elite = candidates[:n_top]
        return elite

    def compute_new_mean(self, e_candidates):
        
        new_means = np.mean(e_candidates, axis=0)
        new_means = np.tile(new_means,(self.n_samples,1))
        return new_means


    def compute_new_std(self, e_candidates):
        eps = 1e-3
        new_std = np.std(e_candidates, ddof=1 ,axis=0) + eps
        new_means = np.tile(new_std,(self.n_samples,1))
        return new_std

    def evaluate_fitness(self, candidates, func=0):
        if func == 0:
            return SphereFunction(candidates)
        else:
            return rastriginFunction(candidates)

    def plot_candidates(self, candidates, func=0, min_val=-5, max_val=5):

        if func == 0:
            plot_visualize_sphere(1000, candidates, min_val, max_val)
        else:
            plot_visualize_rastrigin(1000, candidates, min_val, max_val)
        


# NES
class NES:
    def __init__(self):
        super().__init__()

        # Initialize mean and standard deviation
        #self.theta_mean = np.zeros((n_samples, dim_theta))
        self.theta_mean  = np.random.uniform(min_val, max_val, (n_samples, dim_theta))
        self.theta_std = np.random.uniform(max_val-1, max_val, (n_samples, dim_theta))
        self.n_samples = n_samples
        self.t = n_iterations
        self.top_p = top_p
        self.fit_gaussian()

    def fit_gaussian(self):
        # theta is actualy the population sampled from the distribution
        self.theta = np.random.normal(self.theta_mean, self.theta_std)
        #self.theta = np.clip(theta, min_val, max_val)

        

    def generation(self, function=0):
        # Sample n_sample candidates from N(theta)
        mean_fitness = []
        best_fitness = []
        worst_fitness = []
        I = np.identity(dim_theta*2)
        for i in tqdm(range(0, self.t)):
            #0 --> Sphere; 1 --> Rasti
            fitness = self.evaluate_fitness(self.theta, function)

            mean_fitness.append(np.mean(fitness))
            best_fitness.append(np.min(fitness))
            worst_fitness .append(np.max(fitness))

            
            if plot == 1:
                self.plot_candidates(self.theta, function, min_val, max_val)
                plt.pause(pause)


            # Compute the two gradient separately 
            Dlog_mean = self.compute_mean_grad(self.theta)
            Dlog_std = self.compute_std_grad(self.theta)

            Dlog = np.concatenate((Dlog_mean, Dlog_std), axis=1)

            Dj = np.mean(Dlog * np.array([fitness]).T, axis=0)

            F = np.zeros((Dlog.shape[1], Dlog.shape[1]))
        
            for i in range(Dlog.shape[0]):
                F = F + np.outer(Dlog[i,:], Dlog[i,:])
            
            F = F / self.n_samples
            F = F + I * 1e-5

            theta = np.concatenate((self.theta_mean, self.theta_std), axis=1)

            Theta = theta - alpha * np.dot(np.linalg.inv(F), Dj)

            self.theta_mean = Theta[:, :int(Theta.shape[1]/2)]
            self.theta_std = Theta[:, int(Theta.shape[1]/2):]
            self.fit_gaussian()

            if plot == 1:
                plt.close("all")
        if plot == 1:        
            plt.show()


        print("mean fitness level")
        print(mean_fitness)


        plt.plot(mean_fitness)
        plt.show()

        return mean_fitness, best_fitness, worst_fitness
        
        




    def compute_mean_grad(self, e_candidates):
        eps = 1e-6
        N = e_candidates - self.theta_mean
        D = self.theta_std ** 2
        return N/D

    def compute_std_grad(self, e_candidates):
        eps = 1e-6
        N = (e_candidates - self.theta_mean)**2 - self.theta_std**2
        D = self.theta_std ** 3
        return N/D


    def evaluate_fitness(self, candidates, func=0):
        if func == 0:
            return SphereFunction(candidates)
        else:
            return rastriginFunction(candidates)

    def plot_candidates(self, candidates, func=0, min_val=-5, max_val=5):

        if func == 0:
            plot_visualize_sphere(1000, candidates, min_val, max_val)
        else:
            plot_visualize_rastrigin(1000, candidates, min_val, max_val)




# CMA-ES
class CMA_ES:

    def __init__(self):

        # Initialize mean and standard deviation
        #self.theta_mean = np.zeros((dim_theta))
        self.theta_mean = np.random.uniform(min_val, max_val, (dim_theta))

        theta_std = np.random.uniform(max_val-1, max_val, (dim_theta))
        self.theta_cov = np.diag(theta_std)

        self.n_samples = n_samples
        self.t = n_iterations
        self.top_p = top_p
        self.fit_gaussian()

    def fit_gaussian(self):
        # theta is actually the population sampled from the distribution
        theta = np.random.multivariate_normal(self.theta_mean, self.theta_cov, (n_samples))
        self.theta = np.clip(theta, min_val, max_val)

        

    def generation(self, function=0):
        # Sample n_sample candidates from N(theta)
        mean_fitness = []
        best_fitness = []
        worst_fitness = []

        for i in tqdm(range(0, self.t)):
            #0 --> Sphere; 1 --> Rasti
            fitness = self.evaluate_fitness(self.theta, function)

            mean_fitness.append(np.mean(fitness))
            best_fitness.append(np.min(fitness))
            worst_fitness .append(np.max(fitness))

            couple = list(zip(self.theta, fitness.T))
            sorted_fitness= sorted(couple, key=lambda tup: tup[1])
            elite = self.take_elite(sorted_fitness)
            
            e_candidates = [i[0] for i in elite]

            if plot == 1:
                self.plot_candidates(self.theta, function, min_val, max_val)
            plt.pause(pause)
            self.theta_cov = self.compute_new_cov(e_candidates)
            self.theta_mean = self.compute_new_mean(e_candidates)
            self.fit_gaussian()
    
            
            if plot == 1:
                plt.close("all")
        if plot == 1:        
            plt.show()
        
        plt.plot(mean_fitness)
        plt.show()

        return mean_fitness, best_fitness, worst_fitness
        
        

    def take_elite(self, candidates):
        n_top = int((self.n_samples * self.top_p)/ 100)
        elite = candidates[:n_top]
        return elite

    def compute_new_mean(self, e_candidates):
        
        new_means = np.mean(e_candidates, axis=0)
        return new_means


    def compute_new_cov(self, e_candidates):
        e_candidates = np.array(e_candidates)
        I = np.identity(dim_theta)
        cov = np.zeros((dim_theta, dim_theta))
        for i in range(dim_theta):
            for j in range(dim_theta):
                cov[i,j] = np.sum(((e_candidates[:,i] - self.theta_mean[i]) * (e_candidates[:,j] - self.theta_mean[j])), axis=0)

        return 1/e_candidates.shape[0] * cov + I * 1e-3


    

    def evaluate_fitness(self, candidates, func=0):
        if func == 0:
            return SphereFunction(candidates)
        else:
            return rastriginFunction(candidates)

    def plot_candidates(self, candidates, func=0, min_val=-5, max_val=5):

        if func == 0:
            plot_visualize_sphere(100, candidates, min_val, max_val)
        else:
            plot_visualize_rastrigin(100, candidates, min_val, max_val)
        




# Run the desired algorithm and return the data
def run(alg="CEM", function = 0):
    if alg == "CEM":
        cem = CEM()
        m, b, w = cem.generation(function)
    if alg == "NES":
        nes = NES()
        m, b, w = nes.generation(function)
    if alg == "CMA_ES":
        cma = CMA_ES()
        m, b, w = cma.generation(function)

    return m, b, w


# number of dimensions
dim_theta = 100

# learning rate of NES
alpha = 1e-3

# Population Size
n_samples = 1000

# Elite ratio percentage
top_p = 20

# Number of Generations
n_iterations = 1000

# Range of values
min_val = -5
max_val = 5

# Plot or not (0 = False, 1 = True)
plot = 0

# Number of Runs
runs = 3

# Algorithm to run in { CEM, NES, CMA_ES }
algorithm = "NES"

# Evaluation function ( 0 = Sphere, 1 = Rastrigin)
function = 0

# Plot output frequency
pause = 0.01

mean = []
best = []
worst = []

# Plot samples on contour plot
generateSamples(min_val = min_val, max_val = max_val)

# Run the algorithm runs times and save the results
for i in range(runs):
    m, b, w = run(alg=algorithm, function=function)
    mean.append(m)
    best.append(b)
    worst.append(w)

# Compute the mean over the runs
mean_3 = np.mean(mean, axis=0)
best_3 = np.mean(best, axis=0)
worst_3 = np.mean(worst, axis=0)

# Plot the data
#plt.plot(mean_3, label='mean')
plt.plot(best_3, 'g', label='best')
plt.plot(worst_3, 'r', label='worst')
plt.legend(loc='upper right')
plt.show()


# Save data
np.save(directory + "/" + algorithm + "_" + "mean"  + "_" + str(dim_theta) + "_" + str(n_samples) + "_" + str(n_iterations) + "_" + str(top_p) + "_" + str(alpha), mean_3)
np.save(directory + "/" + algorithm + "_" + "best" + "_" + str(dim_theta) + "_" + str(n_samples) + "_" + str(n_iterations) + "_" + str(top_p) + "_" + str(alpha), best_3)
np.save(directory + "/" + algorithm + "_" + "worst" + "_" + str(dim_theta) + "_" + str(n_samples) + "_" + str(n_iterations) + "_" + str(top_p) + "_" + str(alpha), worst_3)

