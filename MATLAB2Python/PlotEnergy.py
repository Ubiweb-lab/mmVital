import numpy as np

def PlotEnergy(w, Q, r, fs):
    J = len(w) - 1
    # print('J_plot', J)
    e = np.zeros(J + 1)
    for j in range(J + 1):
        e[j] = np.sum(np.abs(w[j])**2)

    return e

# Example usage:
# Assuming you have w from the previous code
# Replace the following line with your actual w, Q, r, and fs values
# N = 200
# J = 3
# w = [np.sin(2 * np.pi * np.arange(1, N+1) / N) for _ in range(J)]
#
# print('w', w)
# Q = 4
# r = 3
# fs = 1000
# energy_values = PlotEnergy(w, Q, r, fs)
# print("shape of Energy Values:", energy_values.shape)
# print("Energy Values:", energy_values)
