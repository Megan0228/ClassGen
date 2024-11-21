import numpy as np
import matplotlib.pyplot as plt

# Define the function for each curve
def curve1(rho_min):
    return 1.8 - 2 * 0.5 * np.log(rho_min / (1 - rho_min))

def curve2(rho_min):
    return 1.8 - 2 * 0.5 * np.log(100 * (rho_min + 0.01) / (1 - (rho_min + 0.01)))

def curve3(rho_min):
    return 1.8 - 2 * 0.5 * np.log(100 * (rho_min + 0.5) / (1 - (rho_min + 0.5)))

# Generate \widehat{\rho}_{\min} values within a valid range for log
rho_min = np.linspace(0.001, 0.999, 1000)

# Calculate y values for each curve
y1 = curve1(rho_min)
y2 = curve2(rho_min)
y3 = curve3(rho_min)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(rho_min, y1, label=r'$1.8 - 2\cdot0.5\log\left(\frac{\widehat{\rho}_{\min}}{1-\widehat{\rho}_{\min}}\right)$', color='b')
plt.plot(rho_min, y2, label=r'$1.8 - 2\cdot0.5\log\left(\frac{100(\widehat{\rho}_{\min}+0.01)}{1-(\widehat{\rho}_{\min}+0.01)}\right)$', color='r')
plt.plot(rho_min, y3, label=r'$1.8 - 2\cdot0.5\log\left(\frac{100(\widehat{\rho}_{\min}+0.5)}{1-(\widehat{\rho}_{\min}+0.5)}\right)$', color='g')

# Add labels and title
plt.xlabel(r'$\widehat{\rho}_{\min}$')
plt.ylabel('$\Delta_{\text{gap}}$')
plt.title('Visualization of the Three Curves with $\widehat{\rho}_{\min}$ as x-axis')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
