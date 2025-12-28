#!/usr/bin/env python3
"""Check GRIB file contents"""
import sys
import pygrib

if len(sys.argv) < 2:
    print("Usage: python check_grib.py <grib_file>")
    sys.exit(1)

grib_file = sys.argv[1]

print(f"Checking GRIB file: {grib_file}")
print("=" * 80)

grbs = pygrib.open(grib_file)
total = len(grbs)
print(f"Total messages/parameters: {total}\n")

grbs.rewind()

# Check for key parameters needed by neural network
key_params = [
    'Temperature',
    'U component of wind',
    'V component of wind',
    'Relative humidity',
    'Geopotential Height',
    'Vertical velocity'
]

print("First 30 parameters:")
print("-" * 80)
for i, grb in enumerate(grbs):
    level_type = grb.typeOfLevel
    level = grb.level
    print(f"{i+1:4d}. {grb.name:45s} {level:6} {level_type:20s}")
    if i >= 29:
        break

print("\n" + "=" * 80)
print("Checking for key parameters needed by neural network:")
print("-" * 80)

grbs.rewind()
found_params = {}
for grb in grbs:
    name = grb.name
    for key in key_params:
        if key in name:
            if key not in found_params:
                found_params[key] = []
            found_params[key].append(f"{grb.level} {grb.typeOfLevel}")

for key in key_params:
    if key in found_params:
        print(f"✓ {key}: {len(found_params[key])} levels")
        # Show first few levels
        for level in found_params[key][:5]:
            print(f"    - {level}")
        if len(found_params[key]) > 5:
            print(f"    ... and {len(found_params[key]) - 5} more")
    else:
        print(f"✗ {key}: NOT FOUND")

grbs.close()
