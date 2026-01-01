
import pickle
from pathlib import Path

data_dir = Path('neural_network/bin/data')
with open(data_dir / 'meteo_params.pkl', 'rb') as f:
    params = pickle.load(f)

hours = set()
for p in params:
    hours.add(p[0])

print("Hours:", sorted(list(hours)))
