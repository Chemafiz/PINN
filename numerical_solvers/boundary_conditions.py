import numpy as np

def v_out_step(theta):
    low = np.random.uniform(0, np.pi / 6)
    high = np.random.uniform(np.pi / 3, np.pi)

    V_out = np.where((theta > low) & (theta < high), 1, -1)
    return V_out

def v_out_sin(theta):
    freq = np.random.randint(1, 40)
    V_out = np.sin(freq * theta)
    return V_out

def v_out_mixed(theta):
    # Losowe współczynniki
    a = np.random.uniform(0.1, 1.0)
    b = 1 - a  # aby a + b = 1
    freq = np.random.randint(1, 6)
    # Losowy przedział dla skoku
    low = np.random.uniform(0, np.pi / 2)
    high = np.random.uniform(np.pi / 2, np.pi)
    V_out = a * np.sin(freq * theta) + b * np.where((theta > low) & (theta < high), 1, -1)
    return V_out
