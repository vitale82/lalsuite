import lalinference

def h5_get_group(h5path,groupname):
    """
    Read group by walking through the given hdf tree
    @param h5path hdf5 object which supports visit method
    @type h5path A h5py.Group or h5py.File object
    @param groupname Name of group to return
    Returns a h5py object or None if it is not found
    """
    return h5path.visititems(lambda name,g: return g if name==groupname else return None)

def h5_get_posterior(h5path):
    return h5_get_group(h5path,name=lalinference.LALInferenceHDF5PosteriorSamplesGroupName)

def h5_get_nested_samples(h5path):
    return h5_get_group(h5path,name=lalinference.LALInferenceHDF5NestedSamplesGroupName)

def h5_group_to_structarray(h5grp,fill_attributes=False):
    """
    Convert the given group to a numpy structured array
    h5grp must contain 1 or more datasets of equal length, which form
    the columns of the returned array.
    If fill_attributes is True, attributes are interpreted as fixed
    parameters whose values are copied into new columns.
    """
    import numpy as np
    # Get the column names
    names=[n for n in h5grp.iterkeys() if isinstance(h5grp[n],h5py.Dataset)]
    N=len(h5grp[names[0]])
    if any( len(h5grp[n])!=N for n in names ):
        raise RuntimeError("Not all datasets in posterior group are same length")
    arr = np.array(np.hstack( h5grp[n] for n in names ), dtype=[(n,np.float64) for n in names])
    if fill_attributes:
        anames=[n for n in h5grp.attrs]
        for n in anames:
            val=h5grp.attrs[n]
            arr = np.hstack((arr, val*np.ones( (N,), dtype=(n,np.float64) ) ))
    return arr


def read_nested_from_hdf5(nested_path_list):
    headers = None
    input_arrays = []
    metadata = {}
    log_noise_evidences = []
    log_max_likelihoods = []
    nlive = []

    def update_metadata(level, attrs, collision='raise'):
        """Updates the sub-dictionary 'key' of 'metadata' with the values from
        'attrs', while enforcing that existing values are equal to those with
        which the dict is updated.
        """
        if level not in metadata:
            metadata[level] = {}
        for key in attrs:
            if key in metadata[level]:
                if attrs[key] != metadata[level][key]:
                    if collision == 'raise':
                        raise ValueError(
                            'Metadata mismtach on level %r for key %r:\n\t%r != %r'
                            % (level, key, attrs[key], metadata[level][key]))
                    elif collision == 'append':
                        if isinstance(metadata[level][key], list):
                            metadata[level][key].append(attrs[key])
                        else:
                            metadata[level][key] = [metadata[level][key], attrs[key]]
                    elif collision == 'ignore':
                        pass
                    else:
                        raise ValueError('Invalid value for collision: %r' % collision)
            else:
                metadata[level][key] = attrs[key]
        return

    for path in nested_path_list:
        with h5py.File(path, 'r') as hdf:
            # walk down the groups until the actual data is reached, storing
            # metadata for each step.
            current_level = 'lalinference'
            group = hdf[current_level]
            update_metadata(current_level, group.attrs)

            if len(hdf[current_level].keys()) != 1:
                raise KeyError('Multiple run-identifiers found: %r'
                               % list(hdf[current_level].keys()))
            # we ensured above that there is only one identifier in the group.
            run_identifier = list(hdf[current_level].keys())[0]

            current_level = 'lalinference/' + run_identifier
            group = hdf[current_level]
            update_metadata(current_level, group.attrs, collision='append')

            # store the noise evidence and max likelihood seperately for later use
            log_noise_evidences.append(group.attrs['log_noise_evidence'])
            log_max_likelihoods.append(group.attrs['log_max_likelihood'])
            nlive.append(group.attrs['number_live_points'])

            # storing the metadata under the posterior_group name simplifies
            # writing it into the output hdf file.
            current_level = 'lalinference/' + run_identifier + '/' + nested_grp_name
            current_level_posterior = 'lalinference/' + run_identifier + '/' + posterior_grp_name
            group = hdf[current_level]
            update_metadata(current_level_posterior, group.attrs)
            # copy the data into memory
            input_data_dict = {}
            for key in group:
                input_data_dict[key] = group[key][...]
            # copy the parameter names and crosscheck with possible previous files
            if headers is None:
                headers = sorted(group.keys())
            elif headers != sorted(group.keys()):
                raise ValueError('Mismatch in input parameters:\n\t%r\n\t\t!=\n\t%r'
                                 % (headers, sorted(group.keys())))
        input_arrays.append(array([input_data_dict[key] for key in headers]).T)

    # for metadata which is in a list, take the average.
    for level in metadata:
        for key in metadata[level]:
            if isinstance(metadata[level][key], list):
                metadata[level][key] = mean(metadata[level][key])

    log_noise_evidence = reduce(logaddexp, log_noise_evidences)
    log_max_likelihood = max(log_max_likelihoods)

    return headers, input_arrays, log_noise_evidence, log_max_likelihood, metadata, nlive, run_identifier

