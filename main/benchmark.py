import random

# Base Benchmark 
MAX_SEG_PIXEL = 0.4587
MAX_SEG_INSTANCE = 0.4786
MAX_SEG_CENTER = 0.5793
MAX_SEG_FDR = 0.5056

# Norm
def norm(max_threshold):
    return random.uniform(0, max_threshold)
