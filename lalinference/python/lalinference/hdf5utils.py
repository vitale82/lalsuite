import lalinference

def get_group(h5path,groupname):
    """
    Read group by walking through the given hdf tree
    @param h5path hdf5 object which supports visit method
    @type h5path A h5py.Group or h5py.File object
    @param groupname Name of group to return
    Returns a h5py object or None if it is not found
    """
    return h5path.visititems(lambda name,g: g if name.rsplit('/')[-1]==groupname else None)

def get_posterior_group(h5path):
    return h5_get_group(h5path,groupname=lalinference.LALInferenceHDF5PosteriorSamplesGroupName)

def get_nested_samples_group(h5path):
    return h5_get_group(h5path,groupname=lalinference.LALInferenceHDF5NestedSamplesGroupName)

def group_to_array(h5grp,fill_attributes=False):
    """
    Convert the given group to a numpy structured ndarray
    h5grp must contain 1 or more datasets of equal length, which form
    the columns of the returned array.
    If fill_attributes is True, attributes are interpreted as fixed
    parameters whose values are copied into new columns.
    """
    import numpy as np
    import h5py
    # Get the column names
    names=[n.rsplit('/')[-1] for n in h5grp.iterkeys() if isinstance(h5grp[n],h5py.Dataset)]
    N=len(h5grp[names[0]])
    if any( len(h5grp[n])!=N for n in names ):
        raise RuntimeError("Not all datasets in posterior group are same length")
    dtype=[(str(n),h5grp[n].dtype) for n in names]
    bigarr=np.vstack( h5grp[n] for n in names )
    if fill_attributes:
        anames=[str(n) for n in h5grp.attrs]
        for n in anames:
            val=h5grp.attrs[n]
            bigarr = np.vstack((bigarr, val*np.ones( (N,)) ) )
            names.append(n)
            dtype.append((n,type(val)))
    return np.array(zip(*bigarr), dtype=dtype)

def load_chain(h5path,groupname,fill_attributes=True):
    """
    Loads the desired group from the h5 path
    Returns pylal.bayespputils.Posterior object
    if fill_attributes is True, attributes are interpreted as fixed
    parameters.
    """
    from pylal import bayespputils as bppu
    from numpy import array
    arr=group_to_array(get_group(h5path,groupname), fill_attributes=fill_attributes)
    params=arr.dtype.names
    p=bppu.Posterior((params,array(zip(*arr))))
    return p

def load_posterior_array(filename,fill_attributes=False):
  """
  Reads the posterior samples group from a given file
  Returns a numpy structured array
  """
  import h5py
  with h5py.File(filename,'r') as h5file:
    return group_to_array( \
      get_group(h5path \
	, groupname=lalinference.LALInferenceHDF5PosteriorSamplesGroupName) \
      , fill_attributes=fill_attributes)
  return None

def load_posterior_from_file(filename):
    """
    Reads the posterior_samples group from a given file
    Returns a bppu.Posterior object
    """
    import h5py
    with h5py.File(filename,'r') as h5file:
        return load_chain(h5file,groupname=lalinference.LALInferenceHDF5PosteriorSamplesGroupName,fill_attributes=True)
    return None