import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Generate example data (x, y, intensity)
x = np.random.rand(100) * 10  # Random x values between 0 and 10
y = np.random.rand(100) * 10  # Random y values between 0 and 10
intensity = np.random.rand(100)  # Random intensity values

# Create a grid for the scatter plot
X, Y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))

# Interpolate intensity values onto the grid
intensity_grid = np.zeros_like(X)
for i in range(len(x)):
    intensity_grid += intensity[i] * np.exp(-((X - x[i])**2 + (Y - y[i])**2) / 2)

# Apply Gaussian filter
sigma = 1  # Standard deviation of the Gaussian kernel
filtered_intensity = gaussian_filter(intensity_grid, sigma=sigma)

# Plot original and filtered intensity
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, y, c=intensity, cmap='viridis')
plt.title('Original Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Intensity')
plt.subplot(1, 2, 2)
plt.imshow(filtered_intensity, cmap='viridis', origin='lower', extent=(0, 10, 0, 10))
plt.title('Filtered Intensity (Gaussian)')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Filtered Intensity')
plt.tight_layout()
plt.show()
