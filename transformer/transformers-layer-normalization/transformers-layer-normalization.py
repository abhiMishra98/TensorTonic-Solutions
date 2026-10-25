import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mu = np.mean(x,axis = -1, keepdims = True)
    
    sigma = np.var(x, axis=-1, keepdims=True)

    num = x - mu
    den = np.sqrt(sigma + eps)

    out = (num/den) * gamma + beta  
    return out
    pass