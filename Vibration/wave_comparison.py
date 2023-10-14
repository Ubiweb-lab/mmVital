import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load data from CSV files
data1 = pd.read_csv('1.csv')
data2 = pd.read_csv('255.CSV')

# Assuming the columns are named 'X', 'Y', and 'Z'
x1, y1, z1 = data1['X'], data1['Y'], data1['Z']
x2, y2, z2 = data2['X'], data2['Y'], data2['Z']

# Combine X, Y, and Z into a single array for simplicity
combined_data1 = np.sqrt(x1**2 + y1**2 + z1**2)
combined_data2 = np.sqrt(x2**2 + y2**2 + z2**2)

# Define a Fourier series function for fitting with limited components
def fourier_series(x, *params):
    a0 = params[0]
    terms = len(params) // 2  # Number of terms in the Fourier series
    result = a0  # DC component
    for i in range(terms):
        result += params[i+1] * np.cos((i+1) * x) + params[i+terms] * np.sin((i+1) * x)
    return result

# Estimate initial parameters using Fourier Transform and select top 8 components
def estimate_initial_params(data, num_terms):
    n = len(data)
    fft_vals = np.fft.fft(data)
    fft_vals_abs = np.abs(fft_vals)  # Calculate absolute values
    print('fft_vals_abs', fft_vals_abs)
    # Sort the array in ascending order
    # Sort the array in descending order
    sorted_fft_vals_abs_desc = np.sort(fft_vals_abs)[::-1]

    print("Sorted fft_vals_abs (descending order):", sorted_fft_vals_abs_desc)
    top_indices = np.argsort(fft_vals_abs)[::-1][:num_terms]  # Get indices of top amplitudes
    print('top_indices', top_indices)
    top_amplitudes = fft_vals_abs[top_indices]  # Get top amplitudes
    print('top_amplitudes', top_amplitudes)
    top_frequencies = np.fft.fftfreq(n)[top_indices]  # Get corresponding frequencies
    print('top_frequencies', top_frequencies)
    #a_coeffs = 2/n * top_amplitudes  # Calculate a coefficients (cosine terms)
    #b_coeffs = 2/n * top_amplitudes  # Calculate b coefficients (sine terms)
    a_coeffs = 2/n * top_amplitudes * np.cos(2 * np.pi * top_frequencies / n)
    print('a_coeffs', a_coeffs)
    b_coeffs = 2/n * top_amplitudes * np.sin(2 * np.pi * top_frequencies / n)
    print('b_coeffs', b_coeffs)
    return np.concatenate(([np.mean(data)], a_coeffs, b_coeffs))  # Include DC component

# Choose the number of terms in the Fourier series (limited to 8)
num_terms = 50

# Estimate initial parameters for the first set of vibration data
initial_params = estimate_initial_params(combined_data1, num_terms)
print('Initial Parameters:', initial_params)

# Generate numerical x values
x_values = np.arange(len(combined_data1))

# Fit the Fourier series to the first set of vibration data using initial parameters
params, _ = curve_fit(fourier_series, x_values, combined_data1, p0=initial_params)

print('Parameters:', params)


# Generate x values for plotting
x_values_plot = np.linspace(min(x_values), max(x_values), 1000)  # Adjust as needed

# Calculate the fitted Fourier series for the plot
fit_plot = fourier_series(x_values_plot, *params)

# Unpack the parameters from the tuple
a0 = params[0]
a_coeffs = params[1:num_terms+1]
b_coeffs = params[num_terms+1:]

# Define the function to represent the fitted Fourier series
def fitted_fourier_series_func(x, a0, *coeffs):
    terms = len(coeffs) // 2
    result = a0  # DC component
    for i in range(terms):
        a_coeff = coeffs[i]
        b_coeff = coeffs[i+terms]
        result += a_coeff * np.cos((i+1) * x) + b_coeff * np.sin((i+1) * x)
    return result

# Print the found fitting function
print("Fitted Fourier Series Function:")
print("fitted_fourier_series(x) = ", end="")
print(f"{params[0]:.2f}", end="")  # Print DC component
for i in range(len(params) // 2):
    a_coeff = params[i+1]
    b_coeff = params[i+len(params) // 2 + 1]
    print(f" + {a_coeff:.2f} * cos({i+1}x) + {b_coeff:.2f} * sin({i+1}x)", end="")

# Example of usage:
x_example = np.linspace(min(x_values), max(x_values), 1000)
fit_example = fitted_fourier_series_func(x_example, *params)
print("\n\nExample of usage:")
print("For x =", x_example[0], ", fitted_fourier_series(x) =", fit_example[0])

# Calculate the predicted values using the fitted Fourier series function
#predicted_values = fitted_fourier_series_func(x_values)  # Use your fitted Fourier series function
predicted_values = fitted_fourier_series_func(x_values, a0, *a_coeffs, *b_coeffs)

# Calculate R-squared
r_squared = r2_score(combined_data1, predicted_values)
print("R-squared:", r_squared)
max_combined_data2 = np.max(combined_data2)
print('Maximum value in combined_data2:', max_combined_data2)



# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_values_plot, fit_plot, label='Fitted Fourier Series (Limited Components)')
plt.plot(combined_data2, label='Current vibration Data')
plt.plot(combined_data1, label='Running car Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
