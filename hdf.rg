import "regent"
local c     = regentlib.c
local hdf5 = terralib.includec(os.getenv("HDF_HEADER") or "hdf5.h")
local coloring   = require("coloring_util")

-- Source: HDF5 I/0 Test Suite
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

task metavalues(r_meta : region(ispace(int1d), meta))
where
  reads(r_meta)
do
  var N : int64 = 0
  for e in r_meta do
    N += r_meta[e].n
    c.printf("r_meta[%d].n = %d\n", e, r_meta[e].n)
  end
  return N
end

task particledata(r_data : region(ispace(int1d), particle))
where
  reads(r_data)
do
  for e in r_data do
    c.printf("r_data[%d].x = %f, r_data[%d].v = %f\n", e, r_data[e].dx, e, r_data[e].v)
  end
end

task toplevel()
  var cores : int32 = 2
  
  var metafile = "particle/particle"
  var r_meta = region(ispace(int1d, cores), meta)
  --attach(hdf5, r_meta.n, metafile, regentlib.file_read_write)
  
  attach(hdf5, r_meta.n, metafile, regentlib.file_read_only)
  acquire(r_meta)
  var N : int64 = metavalues(r_meta)
  __fence(__execution,__block)
  release(r_meta)
  detach(hdf5, r_meta.n)

  var r_particles = region(ispace(int1d, N), particle)
  var c_particles = coloring.create()
  for e in r_meta do
    var Nb : int64 = 0
    var j : int32 = e
    for q = 0, j do
      Nb += r_meta[int1d(q)].n
    end
    c.printf("Nb[%d] = %d\n", e, Nb)
    var p_bounds : rect1d = {Nb+1, Nb+1 + r_meta[e].n}
    coloring.color_domain(c_particles, e, p_bounds)
  end
  var p_particles = partition(disjoint, r_particles, c_particles, ispace(int1d, cores))
  --attach(hdf5, r_particles.{dx,v}, datafile, regentlib.file_read_only)

  for p in p_particles.colors do
    var datafile : int8[200]
    c.sprintf([&int8](datafile), 'particle/particle%03d',p)

    attach(hdf5, (p_particles[p]).{dx,v}, datafile, regentlib.file_read_only)
    acquire((p_particles[p]))
    particledata(p_particles[p])
    release((p_particles[p]))
    detach(hdf5, (p_particles[p]).{dx,v})
  end
  --detach(hdf5, r_particles.{dx,v})
end

regentlib.start(toplevel)
