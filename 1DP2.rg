import "regent"

-- Helper modules to handle PNG files and command line arguments
local Config = require("config")
local coloring   = require("coloring_util")
local Sod = require("sod")
--local Dump = require("dump") -- TODO

-- Some C APIs
local c     = regentlib.c
local sqrt  = regentlib.sqrt(double)
local cmath = terralib.includec("math.h")
local PI = cmath.M_PI

-- Field space for particles
fspace meta
{
  Nc : int64,
  N : int64,
  m : double
}

fspace bins
{
  Np : int64,
  rhogrid : double,
  vgrid : double,
  Pgrid : double
}
fspace particle
{
  x : double,
  v : double,
  xnew : double
}

fspace hist
{
  n : int64,
  mn : double,
  mx : double
}

fspace ghist
{
  rho : double,
  n : int64,
  mn : double,
  mx : double
}

terra dumpdouble(f : &c.FILE, val : double)
  var a : double[1]
  a[0] = val
  c.fwrite(&a, 8, 1, f)
end

terra dumpint32(f : &c.FILE, val : int32)
  var a : int32[1]
  a[0] = val
  c.fwrite(&a, 4, 1, f)
end

terra dumpbool(f : &c.FILE, val : bool)
  var a : int32[1]
  if val==true then
    a[0] = 1
  else 
    a[0] = 0
  end
  c.fwrite(&a, 4, 1, f)
end

terra wait_for(x : int) return 1 end
task block_task(r_image : region(ispace(int1d), particle))
where
  reads writes(r_image)
do
  return 1
end

task Dump(r_hist : region(ispace(int1d), ghist), iter : int32)
where
  reads writes(r_hist)
do
  var filename : int8[1000]
  c.sprintf([&int8](filename), 'Data/rho%d',iter)
  var g = c.fopen(filename,'wb')

  for e in r_hist do
    dumpdouble(g, r_hist[e].rho)
  end
  c.fclose(g)

  return 1
end

terra PBC(x: double)
  while x < 0 do
    --c.printf("%f, left\n", x)
    x = x + 1
  end
  while  x > 1 do
    --c.printf("right\n")
    x = x - 1 
  end
  return x
end


task Advect(r_particles : region(ispace(int1d), particle),
            t : double)
where
  reads(r_particles.{x,v}),
  reads writes(r_particles.xnew)
do
  for p in r_particles do
    r_particles[p].xnew = PBC(r_particles[p].x + r_particles[p].v*t)
    --c.printf("x, xnew = %.3e, %.3e\n", r_particles[p].x, r_particles[p].xnew)
  end
end

task Sort(r_particles : region(ispace(int1d), particle),
          r_hists : region(ispace(int1d), hist),
          boxes : int32)
where
  reads(r_particles.xnew),
  reads writes(r_hists)
do
  for e in r_particles do
    var bin : int1d = r_hists.bounds.lo + r_particles[e].xnew*boxes
    --c.printf("x = %.3e goes in bin %d\n", r_particles[e].xnew, bin)
    r_hists[bin].n = r_hists[bin].n + 1
    if r_hists[bin].mn == 1 then
      r_hists[bin].mn = r_particles[e].xnew
    end
    if r_hists[bin].mx == 0 then
      r_hists[bin].mx = r_particles[e].xnew
    end

    if r_particles[e].xnew < r_hists[bin].mn then    
      r_hists[bin].mn = r_particles[e].xnew
    elseif r_particles[e].xnew > r_hists[bin].mx then
      r_hists[bin].mx = r_particles[e].xnew
    end
  end
  --for e in r_hists do
  --  c.printf("r_hists[%d].{n, mn, mx} = %d, %.3e, %.3e\n", e, r_hists[e].n, r_hists[e].mn, r_hists[e].mx)
  --end
end

task ConsolidateHistograms(r_hists : region(ispace(int1d), hist),
                           r_hist : region(ispace(int1d), ghist), m : double)
-- r_hists is local histograms
-- r_hist is global histograms
-- m is particle mass
where
  reads(r_hists),
  reads writes(r_hist)
do
  fill(r_hist.n, 0)
  fill(r_hist.mn, 1)
  fill(r_hist.mx, 0)

  var boxes = r_hist.volume
  var cells = ispace(int1d, r_hist.bounds.hi - r_hist.bounds.lo + 1)
  for c1 in cells do
    -- c1 is selecting global bin, c2 is selecting local bin. 
    var mn : double = 1
    var mx : double = 0
    var n : int64 = 0
    --c.printf("Global Bin %d\n", c1)
    for c2 in cells do 
      var e = int1d(c1 + c2*boxes)
      
      r_hist[c1].n = r_hist[c1].n + r_hists[e].n
      
      --if r_hists[e].n > 0 then
      --  mn = r_hist[c1].mn
      --end
      --if r_hists[e].n > 0 then
      --  mx = r_hist[c1].mx
      --end

      if r_hists[e].mn < r_hist[c1].mn and r_hists[e].n > 0 then
        r_hist[c1].mn = r_hists[e].mn
      end
      if r_hists[e].mx > r_hist[c1].mx and r_hists[e].n > 0 then
        r_hist[c1].mx = r_hists[e].mx
      end
    end
    --r_hist[c1].n = n
    --r_hist[c1].mn = mn
    --r_hist[c1].mx = mx
  end
  for e in r_hist do
    r_hist[e].rho = m*r_hist[e].n/(r_hist[e].mx-r_hist[e].mn)
    --if e == int1d(0) then
      c.printf("rho[%d] = %.2e * %d / (%.2e - %.2e) = %f\n", e, m, r_hist[e].n, r_hist[e].mx, r_hist[e].mn, r_hist[e].rho)
    --end
  end
end

task metaN(r_meta: region(ispace(int1d), meta))
where
  reads(r_meta)
do
  var N : int64 
  for e in r_meta do
    N = r_meta[e].N
  end
  return N
end

task metaNc(r_meta: region(ispace(int1d), meta))
where
  reads(r_meta)
do
  var Nc : int64 
  for e in r_meta do
    Nc = r_meta[e].Nc
  end
  return Nc
end

task metam(r_meta: region(ispace(int1d), meta))
where
  reads(r_meta)
do
  var m : double 
  for e in r_meta do
    m = r_meta[e].m
  end
  return m
end

task bindata(r_bins: region(ispace(int1d), bins))
where
  reads(r_bins)
do
  for e in r_bins do
    --c.printf("bin[%d]: Np = %d, rho = %f, v = %f, P = %f\n", e, r_bins[e].Np, r_bins[e].rhogrid, r_bins[e].vgrid, r_bins[e].Pgrid)
  end
end

task pdata(r_particles: region(ispace(int1d), particle))
where
  reads(r_particles)
do
  for e in r_particles do
    --c.printf("bin[%d]: Np = %d, rho = %f, v = %f, P = %f\n", e, r_bins[e].Np, r_bins[e].rhogrid, r_bins[e].vgrid, r_bins[e].Pgrid)
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()

  -- Simulation Parameters
  var Tf : double = 0.25           -- Final Time
  var dt : double = Tf/2            -- Initial Timestep
  var out : bool = config.out      -- Output Boolean

  -- Load Metadata 
  var metafile = "particle/particle"
  var r_meta = region(ispace(int1d, 1), meta)
  attach(hdf5, r_meta.{Nc,N,m}, metafile, regentlib.file_read_write)
  acquire(r_meta)
  for e in r_meta do
    c.printf("Nc = %d, N = %d, m = %.3e\n", r_meta[e].Nc, r_meta[e].N, r_meta[e].m)
  end
  var Nc : int32 = metaNc(r_meta)
  var N : int64 = metaN(r_meta)
  var m : double = metam(r_meta)
  c.printf("Nc = %d, N = %d, m = %.3e\n", Nc, N, m)
  release(r_meta)

  -- Load Bin Data
  var binfile = 'particle/particle'
  var r_bins = region(ispace(int1d, Nc), bins)
  attach(hdf5, r_bins.{Np, rhogrid, vgrid, Pgrid}, binfile, regentlib.file_read_write)
  acquire(r_bins)
  bindata(r_bins)
  release(r_bins)
  detach(hdf5, r_bins)

  -- Create a logical region for particles, local and global histograms
  var r_particles = region(ispace(int1d, N), particle)
  var r_hists = region(ispace(int1d, Nc*Nc), hist) --local
  var r_hist = region(ispace(int1d, Nc), ghist) --global
  __fence(__execution, __block)
  c.printf("Regions Created\n")

  -- Create coloring and particle partition based on metadata
  __fence(__execution, __block)
  c.printf("Coloring Particle Partition\n")
  var c_particles = coloring.create()
  var c_hists = coloring.create()
  __fence(__execution, __block)
  c.printf("Coloring Initialized\n")
  for e in r_bins do
    var Nb : int64 = 0
    var j : int32 = e
    for q = 0, j do
      Nb += r_bins[int1d(q)].Np
    end
    var p_bounds : rect1d = {Nb, Nb + r_bins[e].Np-1}
    var h_bounds : rect1d = {Nc*e, Nc*(e+1)-1}
    c.printf("Nb[%d] = %d, bounds = {%d, %d}, hbounds = {%d, %d}\n", e, Nb, p_bounds.lo, p_bounds.hi, h_bounds.lo, h_bounds.hi)
    coloring.color_domain(c_particles, e, p_bounds)
    coloring.color_domain(c_hists, e, h_bounds)
  end
  var p_colors = ispace(int1d, Nc)
  var p_particles = partition(disjoint, r_particles, c_particles, p_colors)
  var p_hists = partition(disjoint, r_hists, c_hists, p_colors)
  __fence(__execution, __block)
  c.printf("Coloring and Partitions Created\n")

  var token : int32 = 0
  var START : double = c.legion_get_current_time_in_micros()
  var Start : double
  var End : double

  -- Load Particle Data
  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  for p in p_colors do
    var datafile : int8[200]
    c.sprintf([&int8](datafile), 'particle/particle%03d', p)
    attach(hdf5, (p_particles[p]).{x,v}, datafile, regentlib.file_read_only)
    acquire((p_particles[p]))
    --pdata((p_particles[p]))
    detach(hdf5, (p_particles[p]))
  end
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Initialization Done in %.3e s\n", (End-Start)*1e-6)

  -- Simulation times
  var tf : double = 2.0
  var ntimes : int32 = 400

  -- Main Loop
  for t = 0, 1 do -- ntimes do

    -- Advect
    __fence(__execution, __block)
    Start = c.legion_get_current_time_in_micros()
    for p in p_colors do
      var datafile : int8[200]
      --c.printf('particle/particle%03d\n', p)
      c.sprintf([&int8](datafile), 'particle/particle%03d', p)
      attach(hdf5, (p_particles[p]).{x,v}, datafile, regentlib.file_read_write)
      acquire((p_particles[p]))
      Advect(p_particles[p], tf*t/ntimes)
      detach(hdf5, (p_particles[p]))
    end
    __fence(__execution, __block)
    End = c.legion_get_current_time_in_micros()
    c.printf("Advect Done in %.3e s\n", (End-Start)*1e-6)

    fill(r_hists.n, 0)
    fill(r_hists.mn, 1)
    fill(r_hists.mx, 0)

    -- Sort
    __fence(__execution, __block)
    Start = c.legion_get_current_time_in_micros()
    for p in p_colors do
      Sort(p_particles[p], p_hists[p], Nc)
    end
    __fence(__execution, __block)
    End = c.legion_get_current_time_in_micros()
    c.printf("Sort Done in %.3e s\n", (End-Start)*1e-6)

    -- Consolidate Hist
    __fence(__execution, __block)
    Start = c.legion_get_current_time_in_micros()
    ConsolidateHistograms(r_hists, r_hist, m)
    __fence(__execution, __block)
    End = c.legion_get_current_time_in_micros()
    c.printf("Consolidation Done in %.3e s\n", (End-Start)*1e-6)

    -- Dump
    Dump(r_hist,t)
    __fence(__execution,__block)
    c.printf("Finished %d\n", t)
  end
  wait_for(token)
  var TS_end = c.legion_get_current_time_in_micros()

  wait_for(token)
  c.printf("Done\n")
  -- Checking values
  c.printf("Checking Values\n")
  c.printf("Total time: %.6f sec.\n", (c.legion_get_current_time_in_micros() - START) * 1e-6)

  for p in p_colors do
    release((p_particles[p]))
    detach(hdf5, (p_particles[p]).{x,v})      
  end


  --Dump(r_particles)
  detach(hdf5, r_bins.{Np,rhogrid,vgrid,Pgrid})
  detach(hdf5, r_meta.{Nc,N,m})
end

regentlib.start(toplevel)
