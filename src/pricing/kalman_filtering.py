import numpy as np
import matplotlib.pyplot as plt 

# Initial State (position and velocity)
x = np.array([0, 1])

# State Transition Matrix
A = np.array([[1, 1],
              [0, 1]])

# Measurement Matrix
H = np.array([[1, 0]])

# Initial Uncertainty
P = np.array([[1000, 0],
              [0, 1000]])

# Process Uncertainty
Q = np.array([[0.1, 0.1],
              [0.1, 0.1]])

# Measurement Uncertainty
R = np.array([[1]])

# Identity Matrix
I = np.eye(2)

# Simulated measurements with some noise
true_positions = np.arange(0, 100, 1)
measurements = true_positions + np.random.normal(0, 1, size=true_positions.shape)

estimated_positions = []

for z in measurements:
    # Prediction
    x = np.dot(A, x)
    P = np.dot(A, np.dot(P, A.T)) + Q
    
    # Update
    Y = z - np.dot(H, x)  # Error
    S = np.dot(H, np.dot(P, H.T)) + R  # Measurement Prediction Uncertainty
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # Kalman Gain
    x = x + np.dot(K, Y)
    P = np.dot((I - np.dot(K, H)), P)
    
    estimated_positions.append(x[0])

# Plotting
plt.figure(figsize=(10,6))
plt.plot(true_positions, label='True Positions')
plt.plot(measurements, 'o', markersize=2, label='Noisy Measurements')
plt.plot(estimated_positions, 'r', linewidth=2, label='Kalman Filter Estimates')
plt.legend()
plt.title('Kalman Filter for Position Estimation')
plt.xlabel('Time')
plt.ylabel('Position')
plt.grid(True)
plt.show()


