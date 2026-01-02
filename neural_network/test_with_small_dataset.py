# coding: utf-8
"""
Simple training script for small PKL dataset (bin/data_test/)
No monkey patching needed - Train now auto-detects nb_cells!
"""

import sys, os

# Set test data path BEFORE importing Train
from inc.bin_obj import BinObj
BinObj.obj_path = "./bin/data_test"

# Now import after path is set
from train import Train
from inc.model import ModelType, ProblemFormulation
from inc.dataset import MeteoData, FlightsData

########################################################################
# TEST TRAINING
########################################################################

if __name__ == "__main__":

    print("\n" + "="*70)
    print("TESTING TRAIN WITH SMALL DATASET")
    print("="*70 + "\n")

    # Load dataset info
    print("Loading dataset info...")
    meteo_data = MeteoData()
    flights_data = FlightsData()

    print(f"  Cells:   {meteo_data.nb_cells}")
    print(f"  Days:    {meteo_data.nb_days}")
    print(f"  Total flights: {sum(len(f) for f in flights_data.flights_by_cell_day)}")

    # Get cells with flights
    cells_with_flights = []
    for cell_idx in range(meteo_data.nb_cells):
        cell_flights = 0
        for day in range(meteo_data.nb_days):
            idx = cell_idx * meteo_data.nb_days + day
            if idx < len(flights_data.flights_by_cell_day):
                cell_flights += len(flights_data.flights_by_cell_day[idx])
        if cell_flights > 0:
            cells_with_flights.append(cell_idx)
            print(f"  Cell {cell_idx}: {cell_flights} flights")

    if not cells_with_flights:
        print("\n❌ ERROR: No cells with flights found!")
        sys.exit(1)

    # Create trainer - NO MONKEY PATCHING NEEDED!
    # Train.__init__ will auto-detect nb_cells from data
    print("\n" + "="*70)
    print("CREATING TRAINER (auto-detecting nb_cells)")
    print("="*70 + "\n")

    model_dir = "./bin/models/TEST_CLASSIFICATION"
    os.makedirs(model_dir, exist_ok=True)

    train = Train(model_dir, ModelType.CELLS, ProblemFormulation.CLASSIFICATION)

    print(f"✓ Trainer created successfully!")
    print(f"  all_cells: {train.all_cells}")
    print(f"  nb_cells auto-detected: {len(train.all_cells)}")

    # Set up model
    print(f"\n  Setting up model for cells: {cells_with_flights}")
    train.set_trained(cells_with_flights, super_resolution=1, load_weights=False)

    # Quick training
    print("\nStarting quick training (5 epochs)...")
    train.train((0.01, 0.001, 5), use_validation_set=False)

    print("\n✓ Training completed!")

    # Save
    print("Saving model...")
    train.save()

    print("\n" + "="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print(f"  Model saved to: {model_dir}")
    print("="*70 + "\n")
