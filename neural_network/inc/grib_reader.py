import pygrib, tqdm
import numpy as np

# read some cells and params of grid file of known structure
class GribReader:

    gribFile = None
    grbindx  = None

    def __init__(self, gribFile):
        self.gribFile = gribFile
        # Create index for fast parameter selection
        self.grbIndx = pygrib.index(gribFile, 'name', 'typeOfLevel', 'level')

        # Write index to disk for caching (.idx file alongside .grb2)
        # This makes subsequent opens much faster (index is loaded from disk)
        idx_file = gribFile + '.idx'
        import os
        if not os.path.exists(idx_file):
            try:
                self.grbIndx.write(idx_file)
            except Exception:
                pass  # Silently fail if write fails (e.g., read-only filesystem)

    def getInfos(self):
        for grb in pygrib.open(self.gribFile):
            return grb.validDate, grb.distinctLatitudes, grb.distinctLongitudes

    def getGridStructure(self):
        for grb in pygrib.open(self.gribFile):
            #print grb.distinctLatitudes
            #print grb.distinctLongitudes
            #print grb.keys()
            resolutionLat = abs(grb.distinctLatitudes[1]  - grb.distinctLatitudes[0])
            resolutionLon = abs(grb.distinctLongitudes[1] - grb.distinctLongitudes[0])
            originLat = grb.distinctLatitudes[-1] - 0.5*resolutionLat
            originLon = grb.distinctLongitudes[0] - 0.5*resolutionLon
            return originLat, originLon, resolutionLat, resolutionLon

    @staticmethod
    def findClosest(val, vect, latOrLon):
        idx = (np.abs(vect - val)).argmin()
        return idx

    def getValues(self, params, cellsLatLon):
        cells = None
        values = []

        for param in params:
            name, level = param
            for l in level:
                try:
                    selected_grbs = self.grbIndx.select(name=name, typeOfLevel=l[0], level=l[1])
                    assert len(selected_grbs) == 1 # several matching parameters found

                    for grb in selected_grbs:
                        # Assume the grid is the same for each param
                        if not cells and cellsLatLon:

                            #========
                            # TODO: quand j'utiliserai plus de lon dans le training: check lon negatives
                            #
                            #for latLon in cellsLatLon:
                            #    print "lat", latLon[0], grb.distinctLatitudes[self.findClosest(latLon[0], grb.distinctLatitudes, 0)]
                            #    print "lon", latLon[1], grb.distinctLongitudes[self.findClosest(latLon[1], grb.distinctLongitudes, 1)]
                            #========

                            cells = [(self.findClosest(latLon[0], grb.distinctLatitudes, 0), self.findClosest(latLon[1], grb.distinctLongitudes, 1)) for latLon in cellsLatLon]

                        for cell in cells:
                            values += [grb.values[cell[0],cell[1]]]
                except ValueError:
                    pass

        if len(values) != len(params)*len(cellsLatLon):
            # Some parameters couldn't be extracted (missing in GRIB)
            # Return partial data instead of None
            pass
        return values

    def get_values_array(self, params, crops):
        stacks = []

        for param in params:
            name, level = param
            for l in level:
                try:
                    selected_grbs = self.grbIndx.select(name=name, typeOfLevel=l[0], level=l[1])
                    assert len(selected_grbs) == 1 # several matching parameters found

                    for grb in selected_grbs:
                        stack = np.empty(0)
                        for crop in crops:
                            stack = np.concatenate((stack, grb.values[crop[0]:crop[1],crop[2]:crop[3]].flatten()))
                        stacks += [stack]

                except ValueError:
                    pass

        if len(stacks) != len(params):
            print("len(stacks)", len(stacks))
            print("len(params)", len(params))
            return None
        else:
            return np.stack(stacks)


class InMemoryGribReader:
    """
    Load entire GRIB file into memory for fast sequential access.

    This is optimized for HDD storage where random seeks are expensive.
    The file is read sequentially once, then all data is accessed from RAM.

    Memory usage: ~500MB per GRIB file (65 messages × ~8MB each)
    """

    def __init__(self, gribFile):
        """
        Load all GRIB messages into memory.

        Args:
            gribFile: Path to GRIB file
        """
        self.gribFile = gribFile

        # Storage for messages and grid info
        # Key: (name, typeOfLevel, level) -> values array
        self.messages = {}
        self.distinctLatitudes = None
        self.distinctLongitudes = None

        # Load all messages sequentially (single disk pass)
        self._load_all_messages()

    def _load_all_messages(self):
        """Load all GRIB messages into memory with sequential read."""
        grbs = pygrib.open(self.gribFile)

        try:
            for grb in grbs:
                # Get grid info from first message
                if self.distinctLatitudes is None:
                    self.distinctLatitudes = grb.distinctLatitudes
                    self.distinctLongitudes = grb.distinctLongitudes

                # Store message values in memory
                # Key: (name, typeOfLevel, level)
                key = (grb.name, grb.typeOfLevel, grb.level)
                self.messages[key] = grb.values.copy()  # Copy to keep in memory after file closes
        finally:
            grbs.close()

    @staticmethod
    def findClosest(val, vect, latOrLon):
        """Find index of closest value in vector."""
        idx = (np.abs(vect - val)).argmin()
        return idx

    def getInfos(self):
        """Get valid date and grid info."""
        # Return info from first message
        if self.messages:
            first_key = next(iter(self.messages))
            return (None, self.distinctLatitudes, self.distinctLongitudes)  # validDate not stored
        return (None, None, None)

    def getGridStructure(self):
        """Get grid origin and resolution."""
        if self.distinctLatitudes is None or len(self.distinctLatitudes) < 2:
            return None

        resolutionLat = abs(self.distinctLatitudes[1] - self.distinctLatitudes[0])
        resolutionLon = abs(self.distinctLongitudes[1] - self.distinctLongitudes[0])
        originLat = self.distinctLatitudes[-1] - 0.5 * resolutionLat
        originLon = self.distinctLongitudes[0] - 0.5 * resolutionLon
        return originLat, originLon, resolutionLat, resolutionLon

    def getValues(self, params, cellsLatLon):
        """
        Extract values for given parameters and cells (from memory).

        Args:
            params: List of (name, [(typeOfLevel, level), ...]) tuples
            cellsLatLon: List of (lat, lon) tuples

        Returns:
            List of values [param1_cell1, param1_cell2, ..., param2_cell1, ...]
        """
        cells = None
        values = []

        for param in params:
            name, level = param
            for l in level:
                key = (name, l[0], l[1])
                if key not in self.messages:
                    continue  # Parameter not found in file

                values_array = self.messages[key]

                # Map cells to grid indices (first time only)
                if cells is None and cellsLatLon:
                    cells = [
                        (self.findClosest(lat, self.distinctLatitudes, 0),
                         self.findClosest(lon, self.distinctLongitudes, 1))
                        for lat, lon in cellsLatLon
                    ]

                # Extract values for each cell
                for cell in cells:
                    values.append(values_array[cell[0], cell[1]])

        return values

