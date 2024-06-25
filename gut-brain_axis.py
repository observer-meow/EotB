import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class GutBrainAxisSimulation:
    def __init__(self):
        # Initial conditions
        self.microbiota = 100  # Arbitrary units
        self.serotonin = 50  # Arbitrary units
        self.brain_activity = 10  # Arbitrary units

        # Parameters
        self.microbiota_growth_rate = 0.1
        self.microbiota_decay_rate = 0.05
        self.serotonin_production_rate = 0.2
        self.serotonin_decay_rate = 0.1
        self.brain_activity_increase_rate = 0.15
        self.brain_activity_decay_rate = 0.1

    def derivatives(self, y, t):
        microbiota, serotonin, brain_activity = y

        dmicrobiota_dt = (self.microbiota_growth_rate * microbiota
                          - self.microbiota_decay_rate * microbiota)

        dserotonin_dt = (self.serotonin_production_rate * microbiota
                         - self.serotonin_decay_rate * serotonin)

        dbrain_activity_dt = (self.brain_activity_increase_rate * serotonin
                              - self.brain_activity_decay_rate * brain_activity)

        return [dmicrobiota_dt, dserotonin_dt, dbrain_activity_dt]

    def simulate(self, time_span):
        initial_conditions = [self.microbiota, self.serotonin, self.brain_activity]
        t = np.linspace(0, time_span, num=1000)
        solution = odeint(self.derivatives, initial_conditions, t)
        return t, solution

    def plot_results(self, t, solution):
        plt.figure(figsize=(12, 8))
        plt.plot(t, solution[:, 0], label='Gut Microbiota')
        plt.plot(t, solution[:, 1], label='Serotonin')
        plt.plot(t, solution[:, 2], label='Brain Activity')
        plt.xlabel('Time')
        plt.ylabel('Concentration/Activity')
        plt.title('Gut-Brain Axis Simulation')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_simulation(self, time_span=100):
        t, solution = self.simulate(time_span)
        self.plot_results(t, solution)


# Run the simulation
simulation = GutBrainAxisSimulation()
simulation.run_simulation()


# Perturb the system
def perturb_system(simulation, perturbation_type):
    if perturbation_type == 'antibiotic':
        simulation.microbiota *= 0.5  # Reduce microbiota by 50%
    elif perturbation_type == 'probiotic':
        simulation.microbiota *= 1.5  # Increase microbiota by 50%
    elif perturbation_type == 'stress':
        simulation.brain_activity *= 1.5  # Increase brain activity by 50%


# Run simulations with perturbations
perturbations = ['antibiotic', 'probiotic', 'stress']

for perturbation in perturbations:
    print(f"\nSimulating effect of {perturbation}")
    sim = GutBrainAxisSimulation()
    perturb_system(sim, perturbation)
    sim.run_simulation()
