from .models import ViPLink, TempSpecEncoder
from .cca import MultiScaleCCA
from .inference import encode_video, encode_ppg, cosine_similarity_matrix

__all__ = ['ViPLink', 'TempSpecEncoder', 'MultiScaleCCA',
           'encode_video', 'encode_ppg', 'cosine_similarity_matrix']
__version__ = '1.0.0'
