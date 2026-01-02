# Model Improvement Plan: Thermodynamic Parameters

## Objective
Enhance the PyParaglide neural network model to better predict thermal conditions and flight safety by incorporating advanced thermodynamic parameters. Currently, the model excels at dynamic (wind-driven) forecasting but lacks explicit inputs for thermal potential and instability.

## 1. Targeted New Parameters
We aim to extract and utilize the following parameters from GFS datasets. These correspond to specific GRIB2 variables.

| Parameter | GFS Variable (GRIB) | Purpose | Priority |
| :--- | :--- | :--- | :--- |
| **PBLH** | Planetary Boundary Layer Height | Defines the "ceiling" for thermal flights. Critical for XC potential. | **High** |
| **TCDC** | Total Cloud Cover | Determines solar heating. 100% cover kills thermals. | **High** |
| **CAPE** | Convective Available Potential Energy | Indicator of thunderstorm risk and explosive lift. | **Medium** |
| **Lifted Index** | Lifted Index (Surface to 500mb) | General atmospheric instability measure. | **Medium** |
| **CIN** | Convective Inhibition | Energy required to initiate convection (the "cap"). | **Low** |

## 2. Implementation Roadmap

### Phase 1: GFS Data Extraction (Ingest)
**Goal:** Modify the GRIB reader to identify and extract the new variables.

*   **File:** `neural_network/inc/grib.py` (or `pyparaglide/data/grib_reader.py` if modernized)
*   **Action:**
    *   Update the `GribReader` class or configuration to look for `Planetary_Boundary_Layer_Height_surface`, `Total_cloud_cover_entire_atmosphere`, `Convective_available_potential_energy_surface`, etc.
    *   Currently, the code likely filters for specific levels (isobaric). Need to ensure "Surface" level variables are captured.
    *   *Note:* Ensure `pygrib` or `cfgrib` logic handles these surface-level fields correctly alongside the 3D pressure-level grids.

### Phase 2: Dataset Pipeline (Processing)
**Goal:** Propagate new data through the building process into `.pkl` files.

*   **File:** `neural_network/inc/dataset.py` (MeteoData class)
*   **Action:**
    *   The `MeteoData` class structures data into `wind`, `humidity`, and `other`.
    *   Decide where to put the new parameters.
        *   **Option A:** Expand `other` (currently 3 dimensions: Pressure, Temp, ?). This is easiest but might dilute the data.
        *   **Option B (Recommended):** Create a new input tensor category `thermo` for surface-level scalar values (PBLH, CAPE, TCDC) that don't depend on altitude layers like wind does.
    *   Update `scripts/build_dataset.py` to process these new fields during the `build` command.

### Phase 3: Model Architecture (Neural Network)
**Goal:** Modify the TensorFlow model to accept and learn from the new inputs.

*   **File:** `neural_network/inc/model.py`
*   **Action:**
    *   **Input Layer:** Add a new input head `input_thermo` (shape: `(nb_cells, 3, nb_thermo_params)`).
    *   **Architecture:**
        *   Inject this new input into the `flyability_block`.
        *   Currently, `flyability_block` takes `wind`, `other`, `rain`. Add `thermo`.
        *   *Hypothesis:* PBLH should strongly correlate with "flyability" (binary classification) and "distance" (regression).
    *   **Code Update:** Update `get_flyability_block` concatenation logic.

### Phase 4: Training Loop
**Goal:** Pass the new numpy arrays to the model during training.

*   **File:** `neural_network/train.py`
*   **Action:**
    *   In `Train.__loadTrainingData()`: Load the new `thermo` matrix.
    *   In `Train.__get_X()`: Add the new data to the returned list `all_X` at the correct index matching the model's input definition.
    *   **Normalization:** Important! PBLH (0-3000m) and CAPE (0-2000+) have different ranges than Temperature. Compute and apply normalization (Mean/Std) for these new fields similar to `normalization_mean_other`.

## 3. Verification Plan
1.  **Unit Test Data Extraction:** Run `check_grib.py` (or create a script) to verify PBLH/CAPE values are not all zeros or NaNs for a sample GFS file.
2.  **Shape Check:** Verify the `.pkl` files have increased size/dimensions.
3.  **Training Dry Run:** Run `pyparaglide train --cell 1 --epochs 1` to ensure no shape mismatch errors in TensorFlow.
4.  **Correlation Check:** (Optional) Before training, plot PBLH vs. Number of Flights to confirm the physical correlation exists in the dataset.

## 4. Migration Notes
*   **Backward Compatibility:** Old trained models (weights.h5) will **NOT** work with the new architecture. This is a breaking change requiring a full retrain (Weather -> Spots).
*   **Version Bump:** Increment model version to `3.0.0` or `2.1.0`.
