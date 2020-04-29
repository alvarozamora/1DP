import "regent"
local c     = regentlib.c
--local mpi = terralib.includec('/share/software/user/open/openmpi/4.0.3/include/mpi.h')
--local hdf5 = terralib.includec('/share/software/user/open/hdf5/1.10.6/include/hdf5.h', '/share/software/user/open/openmpi/4.0.3/include/mpi.h')
local hdf5 = terralib.includec(os.getenv("HDF_HEADER") or "hdf5.h")

-- there's some funny business in hdf5.h that prevents terra from being able to
--  see some of the #define's, so we fix it here, and hope the HDF5 folks don't
--  change the internals very often...
hdf5.H5F_ACC_TRUNC = 2
hdf5.H5T_STD_I32LE = hdf5.H5T_STD_I32LE_g
hdf5.H5T_STD_I64LE = hdf5.H5T_STD_I64LE_g
hdf5.H5T_IEEE_F64LE = hdf5.H5T_IEEE_F64LE_g
hdf5.H5P_DEFAULT = 0


fspace meta
{
  n : int64
}

fspace particle
{
  dx : double,
  v : double
}

task printvalues(r_meta : region(ispace(int1d), meta))
where
  reads(r_meta)
do
  for e in r_meta do
    c.printf("r_meta[%d].n = %d\n", e, r_meta[e].n)
  end
end

task toplevel()
  var cores : int32 = 2
  
  var metafile = "particle/particle"
  var r_meta = region(ispace(int1d, cores), meta)
  --attach(hdf5, r_meta.n, metafile, regentlib.file_read_write)
  
  attach(hdf5, r_meta.n, metafile, regentlib.file_read_only)
  acquire(r_meta)
  printvalues(r_meta)
  __fence(__execution,__block)
  release(r_meta)
  detach(hdf5, r_meta.n)
end
regentlib.start(toplevel)
