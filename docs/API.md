# PyParaglide API Reference

## CLI

```python
from pyparaglide.cli import app

# Run CLI programmatically
app()
```

## Configuration

```python
from pyparaglide.config import Settings, get_settings

# Get settings singleton
settings = get_settings()

# Access configuration
print(f"Training dates: {settings.training_dates}")
print(f"BBox: {settings.bbox}")
print(f"GFS dir: {settings.gfs_dir}")
```

### Settings Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `training_dates` | `str` | Date ranges (format: `YYYY-MM-DD:YYYY-MM-DD,...`) |
| `bbox` | `str` | Bounding box (format: `lat_min,lat_max,lon_min,lon_max`) |
| `gfs_dir` | `Path` | GFS data directory |
| `pkl_dir` | `Path` | PKL dataset directory |
| `flights_dir` | `Path` | Flight data directory |
| `models_dir` | `Path` | Model weights directory |
| `output_dir` | `Path` | Forecast output directory |
| `min_flights_per_spot` | `int` | Minimum flights for spot clustering |
| `spot_cluster_distance_km` | `float` | Spot clustering radius |

## Models

### ModelCells

```python
from pyparaglide.models import ModelCells, ProblemFormulation

# Create model
model = ModelCells.create_model(
    problem_formulation=ProblemFormulation.CLASSIFICATION,
    nb_cells=10,
    wind_dim=8,
    other_dim=45,
    humidity_dim=2,
    nb_altitudes=5,
    super_resolution=1,
)

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train
model.fit(X_train, Y_train, epochs=55, batch_size=32)

# Predict
predictions = model.predict(X_test)
```

### Output Names

```python
output_names = ModelCells.output_names()
# ['flown 1000', 'flown 900', ..., 'flown of rain 600']
```

### ModelSpots

```python
from pyparaglide.models import ModelSpots, ProblemFormulation

cells_data = {
    0: {"spots": [0, 1, 2]},  # Cell 0 has 3 spots
    1: {"spots": [3, 4]},     # Cell 1 has 2 spots
}

model = ModelSpots.create_model(
    problem_formulation=ProblemFormulation.CLASSIFICATION,
    cells_data=cells_data,
    wind_dim=8,
    other_dim=45,
    humidity_dim=2,
    nb_altitudes=5,
    initialization={"date_factor": np.array([[1.275]]), "dow_factor": np.array([[1.0] * 7])},
)
```

## Training

### Trainer

```python
from pyparaglide.training import Trainer
from pyparaglide.models import ModelType, ProblemFormulation

trainer = Trainer(
    data_dir="data/pkl",
    model_type=ModelType.CELLS,
    problem_formulation=ProblemFormulation.CLASSIFICATION,
    models_dir="data/models",
)

# Prepare data
X, Y = trainer.prepare_data(cells=[0, 1, 2], super_resolution=1)

# Create model
trainer.create_model(cells=[0, 1, 2], super_resolution=1)

# Train
history = trainer.train(
    X=X,
    Y=Y,
    lr_init=0.008,
    lr_end=0.0007,
    nb_epochs=55,
    batch_size=32,
    use_validation_set=True,
)

# Save weights
trainer.save_weights()
```

## Dataset

### Dataset

```python
from pyparaglide.data import Dataset

dataset = Dataset("data/pkl")

# Access metadata
print(f"Days: {dataset.nb_days}")
print(f"Cells: {dataset.nb_cells}")

# Get data by cells
X = dataset.get_meteo_matrix(cells=[0, 1], params=dataset.params_other[0])

# Get date/dow
dates = dataset.get_dow()  # (nb_days, 7)
normalized_dates = dataset.get_date()  # (nb_days, 1)

# Get flights by altitude
outputs = dataset.get_flights_by_altitude(
    cells=[0, 1],
    nb_altitudes=5,
    super_resolution=1,
    regression=False,
)
# Returns: [flown, crossed, wind_flown, humidity_flown]
```

### Normalization

```python
from pyparaglide.data import Normalization, compute_normalization_coeffs, apply_normalization

# Compute normalization
data = np.random.randn(100, 45).astype(np.float32)
mean, std = compute_normalization_coeffs(data)

# Apply normalization
normalized = apply_normalization(data, mean, std)

# Save/load normalization
norm = Normalization(
    other_mean=mean,
    other_std=std,
    humidity_mean=np.array([0.0]),
    humidity_std=np.array([1.0]),
)
norm.save("normalization.pkl")
loaded = Normalization.load("normalization.pkl")
```

## Inference

### GribReader

```python
from pyparaglide.inference import GribReader

# Open GRIB file
reader = GribReader("data/gfs/anl/20240801/gfs.t00z.pgrb2.0p25.f000")

# Get parameter by name and level
temp_850 = reader.get_param("Temperature", level=850, type_of_level="isobaricInhPa")

# Get lat/lon grids
lats, lons = reader.get_lat_lon_grid()

# Get all parameter names
params = reader.list_params()
```

### Forecaster

```python
from pyparaglide.inference import Forecaster

forecaster = Forecaster(
    model_path="data/models/cells.weights.h5",
    normalization_path="data/models/normalization.pkl",
    gfs_dir="data/gfs/anl",
)

# Generate forecast for date
forecast_data = forecaster.predict(date="2024-08-01", hour=0)

# Save forecast
forecaster.save_forecast(forecast_data, "output/forecast_20240801.json")
```

## Data Processing

### DatasetBuilder

```python
from pyparaglide.preprocessing import DatasetBuilder

builder = DatasetBuilder(
    bbox=(45, 47, 13, 15),
    start_date=date(2024, 6, 1),
    end_date=date(2024, 8, 31),
    gfs_dir="data/gfs/anl",
    flights_dir="data/flights",
    output_dir="data/pkl",
)

# Build dataset
stats = builder.build()

print(f"Cells: {stats['cells']}")
print(f"Days: {stats['days']}")
```

### GFSDownloader

```python
from pyparaglide.downloads import GFSDownloader

downloader = GFSDownloader(
    gfs_dir="data/gfs/anl",
    start_date=date(2024, 8, 1),
    end_date=date(2024, 8, 31),
    hours=[6, 12, 18],
)

# Download
stats = downloader.download()
print(f"Downloaded: {stats['total_files']} files")
```

## Enums

### ProblemFormulation

```python
from pyparaglide.models import ProblemFormulation

# Use in model creation
model = ModelCells.create_model(
    problem_formulation=ProblemFormulation.CLASSIFICATION,  # or REGRESSION
    ...
)
```

### ModelType

```python
from pyparaglide.models import ModelType

# Use in trainer
trainer = Trainer(
    model_type=ModelType.CELLS,  # or SPOTS
    ...
)
```

## Custom Layers

All custom layers are in `pyparaglide.models.layers`:

- `WindBlockCells` — Wind terrain adjustment
- `WindBlockSpots` — Spot-specific wind processing
- `WindFlyabilityBlock` — Wind-based flyability
- `HumidityFlyabilityBlock` — Rain-based flyability
- `FlyabilityBlock` — Combined flyability
- `CrossabilityBlock` — Cross-country potential
- `PopulationBlock` — Pilot behavior model

```python
from pyparaglide.models.layers import PopulationBlock
from pyparaglide.models import ProblemFormulation

var_date_factor = tf.Variable(np.array([[1.275]], dtype=np.float32))
var_dow_factor = tf.constant(np.array([[1.0] * 7], dtype=np.float32))

layer = PopulationBlock(
    problem_formulation=ProblemFormulation.CLASSIFICATION,
    var_date_factor=var_date_factor,
    var_dow_factor=var_dow_factor,
    super_resolution=1,
)
```
