import numpy as np
from scipy.ndimage import center_of_mass


def intensidad_promedio(img):
    """Promedio de valores de píxeles (nivel de gris)."""
    return np.mean(img)

def varianza_intensidad(img):
    """Varianza de los píxeles."""
    return np.var(img)

def densidad_superior(img):
    """Porcentaje de tinta en la mitad superior."""
    total = np.sum(img)
    return np.sum(img[:14, :]) / total if total > 0 else 0

def simetria_vertical(img):
    """Diferencia entre la imagen y su reflejo vertical."""
    simetria = np.abs(img - np.fliplr(img)).sum()
    return simetria / img.size

def centro_masa_vertical(img):
    """Ubicación vertical del centro de masa (valor entre 0 y 28)."""
    return center_of_mass(img)[0]
