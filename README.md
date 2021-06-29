# Labrad utils
Some utilities for dealing with labrad (or more general stuff).

## List of general utils:
- **labrad_hdf5_to_ndarray** : Load a hdf5 file and return the data (numpy.array) and columns of labels (list)
- **create_labrad_hdf5file** : Create a labrad hdf5 file from ndarray.
- **labrad_hdf5_get_parameters** : Get parameter settings (e.g., ferquency, time constant added by DV.add_parameters()) from a labrad hdf5 file
- **get_parameters_of_func** : Get a dictionary of paramteres of the function.
- **write_meas_parameters** : Write measurement parameters to txt file and labrad hdf5 file.


