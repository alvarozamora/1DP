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
fspace particle
{
   x  : double,
   vx : double,
   t : double,
   --m : double,   
}

-- Field space for collision time ledgers
fspace ledger
{
  t : double,
}

-- Field space for top ledgers
fspace topledger
{
  t : double,
  col : int1d,
  p : int1d
}

terra uniform(data : &c.drand48_data)
  var flip : double[1]
  --c.srand48_r(c.legion_get_current_time_in_nanos(), data)
  c.drand48_r(data, [&double](flip))
  return flip[0]
end

terra normal(data : &c.drand48_data)
  var U1 : double[1]
  var U2 : double[1]
  
  --c.srand48_r(c.legion_get_current_time_in_nanos(), data)
  c.drand48_r(data, [&double](U1))
  c.drand48_r(data, [&double](U2))
 
  var g : double
  g = cmath.sqrt(-2*cmath.log(U1[0]))*cmath.cos(2*PI*U2[0])

  --var g : double[2]
  --g[0] = sqrt(-2*cmath.log(U[0]))*cmath.cos(2*PI*U[1])
  --g[1] = sqrt(-2*cmath.log(U[0]))*cmath.sin(2*PI*U[1])

  return g
end

task Initialize(r_particles : region(ispace(int1d), particle),
                r_rng : region(ispace(int1d), c.drand48_data[1]),
                N : int64, D : double, sod: Sod)
where
  reads writes(r_particles),
  reads writes(r_rng)
do
  -- Initialize Random Number Generator
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end
  c.srand48_r(c.legion_get_current_time_in_nanos(), data)

  -- Initialize Sod Shock Tube Problem
  var NL : int64 = N*(sod.pL/(sod.pL + sod.pR))
  var NR : int64 = N*(sod.pR/(sod.pL + sod.pR))
  if (NL + NR) < N then
    NR += 1
  end

  for e in r_particles do
    if e < int1d(NL) then
      r_particles[e].x = uniform(data)*(0.5/NL - D) + 0.5*[double](e)/[double](NL) + D/2
      r_particles[e].vx = normal(data)*cmath.sqrt(2*sod.PL/sod.pL)
      r_particles[e].t = 0.0
    else
      r_particles[e].x = uniform(data)*(0.5/NR - D) + 0.5*[double](int64(e)-NL)/[double](NR) + 0.5 + D/2
      r_particles[e].vx = normal(data)*cmath.sqrt(2*sod.PR/sod.pR)
      r_particles[e].t = 0.0
    end
    --c.printf("Particle[%d] at x = %.5f with v = %.5f\n", e, r_particles[e].x, r_particles[e].vx)
  end    
  return 1
end


task Fill_Local_Ledgers(r_ledger : region(ispace(int1d), ledger),
               r_particles : region(ispace(int1d), particle),
               N : int64, D : double)
where 
  reads writes (r_ledger),
  reads (r_particles)
do

  for e in r_ledger do
    var l : int1d = e - 1
    var r : int1d = e

    var lx : double 
    var lv : double 
    if l == int1d(-1) then 
      lx = -D/2
      lv = 0
    else
      lx = r_particles[l].x - r_particles[r].vx*r_particles[r].t
      lv = r_particles[l].vx
    end

    var rx : double 
    var rv : double 
    if r == int1d(N) then
      rx = 1.0 + D/2
      rv = 0
    else
      rx = r_particles[r].x - r_particles[r].vx*r_particles[r].t
      rv = r_particles[r].vx
    end

    r_ledger[e].t = (rx - lx - D)/(lv - rv)
    if r_ledger[e].t <= 0 then
      r_ledger[e].t = 1e6
    end
  end
  return 1
end

task Consolidate_Local_Ledgers(r_ledger : region(ispace(int1d), ledger),
                r_top : region(ispace(int1d), topledger),
                n : int32, col: int1d)
where 
  reads writes (r_top),
  reads (r_ledger)
do
  fill(r_top.t, 1e6)
  fill(r_top.p, 1e6)
  fill(r_top.c, 1e6)

  for e in r_ledger do
    var continue : bool = true
    var k : int1d = col*n
  
    while continue==true do

      -- If current time is shorter than top[k], stop while, push back all times, and set top[k] to current time
      if r_ledger[e].t < r_top[k].t then

        -- Stop while loop
        continue = false

        -- Temp Variable
        var top : double = r_ledger[e].t
        var topp : int1d = e
        var topc : int1d = col

        -- Pushback Times
        for j = int32(k), int32((col+1)*n) do

          -- Temp Variable
          var top2 : double = r_top[j].t
          var topp2 : double = r_top[j].p
          var topc2 : double = r_top[j].c

          r_top[j].t = top
          top = top2
        end

      -- Otherwise, continue
      else
        k += 1
      end
      if k == (col+1)*n then
        continue = false
      end
    end
  end
  return 1
end

task Update_Global_Ledger(r_global : region(ispace(int1d), topledger),
                          r_local : region(ispace(int1d), topledger),
                          k : int32, count : int32)
where 
  reads writes(r_global),
  reads (r_local)
do
  fill(r_global.t, 1e6)
  for e in r_local do
    var continue : bool = true
    var idx : int1d = 0

    if count < k then
      k = count
    end

    while continue==true do

      -- If current time is shorter than top[idx], stop while, push back all times, and set top[idx] to current time
      if r_local[e].t < r_global[idx].t then

        -- Stop while loop
        continue = false

        -- Temp Variable
        var top : double = r_local[e].t
        var topp : int1d = r_local[e].p
        var topc : int1d = r_local[e].c

        -- Pushback Times
        for j = int32(idx), k do

          -- Temp Variable
          var top2 : double = r_global[j].t
          var topp2 : int1d = r_global[j].p
          var topc2 : int1d = r_global[j].c

          r_global[j].t = top
          r_global[j].p = topp
          r_global[j].c = topc

          top = top2
          topp = topp2
          topc = topc2
        end

      -- Otherwise, continue
      else
        idx += 1
        if idx == int1d(k) then
          continue = false
        end
      end

    end
  end
end


task Last_Time(r_local : region(ispace(int1d), topledger), k : int32)
where
  reads(r_local)
do
  var Min : double = 1e6
  for e in r_local do
    if (int32(e)%k==(k-1)) then 
      if r_local[e].t < Min then
        Min = r_local[e].t
      end      
    end
  end
  c.printf("Min = %.3e\n", Min)
  var count : int32 = 0
  for e in r_local do
    if r_local[e].t <= Min then
      count += 1
    end      
  end
  c.printf("Count = %d\n", count)
  return count
end

task Collision(r_ledger : region(ispace(int1d), ledger),
               r_particles : region(ispace(int1d), particle),
               N : int64, D : double, e : int1d)
where
  reads writes (r_ledger),
  reads (r_particles)
do

end



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


task Dump(r_grid : region(ispace(int1d), particle), iter : int32)
where
  reads writes(r_grid)
do
  var filename : int8[1000]
  c.sprintf([&int8](filename), 'Grid%d',iter)
  var g = c.fopen(filename,'wb')

  for e in r_grid do
    --dumpdouble(g, r_grid[e].z)
  end
  __fence(__execution, __block)
  for e in r_grid do
    --dumpdouble(g, r_grid[e].vy)
  end
  for e in r_grid do
    --dumpdouble(g, r_grid[e].vx)
  end
  for e in r_grid do
    --dumpdouble(g, r_grid[e].vz)
  end
  __fence(__execution, __block)
  c.fclose(g)

  return 1
end

task Advect(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             L : double, w : double, dt : double)
where
  reads writes(r_particles.{x,vx}),
  reads writes(r_rng)
do
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end

  for p in r_particles do
    --c.printf("Advecting\n")
    var tsim: double = 0
    while tsim < dt do
      var dt1 : double = (dt-tsim)

      --if r_particles[p].z + dt1*r_particles[p].vz < 0 then 
      --  dt1 = (0 - r_particles[p].z)/r_particles[p].vz
      --elseif r_particles[p].z + dt1*r_particles[p].vz > L then
      --  dt1 = (L - r_particles[p].z)/r_particles[p].vz
      --end        
    
      --Advect (either to wall or not)
      r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt1

      --Periodic BC
      --while r_particles[p].x < 0 do
      --  r_particles[p].x = r_particles[p].x + 1.0*w
      --end
      --while r_particles[p].x > w do
      --  r_particles[p].x = r_particles[p].x - 1.0*w
      --end

      --Periodic BC
      if r_particles[p].x < 0 or r_particles[p].x > w then
        r_particles[p].x = cmath.fmod(r_particles[p].x, w)
      end
      --Reflective BC

      if dt1 < dt-tsim then
        --r_particles[p].vz = -r_particles[p].vz
      end
      --Update Tsim Counter
      tsim += dt1
    end
  end
end

task toplevel()
  var config : Config
  config:initialize_from_command()
  var sod : Sod
  sod:initialize_parameters()

  -- Simulation Parameters
  var Tf : double = 0.25           -- Final Time
  var dt : double = Tf/2            -- Initial Timestep
  var out : bool = config.out      -- Output Boolean
  var N : int64 = 1e9              -- Particle Number
  var D : double = 1e-5/N          -- Particle Diameter
  var n : int32 = 5             -- Number of top times, local
  var k : int32 = 30              -- Number of top times, global

  c.printf("N = %d, D = %.3e\n", N, D)

  -- Create a logical region for particles and ledgers
  var r_particles = region(ispace(int1d, N), particle)
  var r_ledger = region(ispace(int1d, N+1), ledger) 
  var r_local = region(ispace(int1d, n*config.p), topledger)
  var r_global = region(ispace(int1d, k), topledger)
  __fence(__execution, __block)
  c.printf("Regions Created\n")

  -- Create an equal partition of the particles
  var p_colors = ispace(int1d, config.p)
  var p_particles = partition(equal, r_particles, p_colors)
  var p_ledger = partition(equal, r_ledger, p_colors)
  var p_local = partition(equal, r_local, p_colors)
  __fence(__execution, __block)
  c.printf("Equal Partitions Created\n")

  -- Create a coloring for halo partition of particles for ledger
  var c_halo = coloring.create()
  for color in p_colors do
    var bounds = p_particles[color].bounds
    var halo_bounds : rect1d = {bounds.lo - 1, bounds.hi + 1}
    coloring.color_domain(c_halo, color, halo_bounds)
  end 
  --Create an aliased partition of particles using coloring for populating ledgers.
  var p_halo = partition(aliased, r_particles, c_halo, p_colors)
  coloring.destroy(c_halo)
  __fence(__execution, __block)
  c.printf("Halo Partition Created\n")

  -- Create a region and partition for random number generators
  var r_rng = region(p_colors, c.drand48_data[1])
  var p_rng = partition(equal, r_rng, p_colors)
  __fence(__execution, __block)
  c.printf("RNGs Initialized\n")

  var token : int32 = 0
  var TS_start = c.legion_get_current_time_in_micros()
  var Start : double
  var End : double

  -- Initialize Particles
  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  for color in p_colors do
    token += Initialize(p_particles[color], p_rng[color], N, D, sod)
  end
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Initialization Done in %.3e s\n", (End-Start)*1e-6)

  -- Initialize Ledgers
  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  for color in p_colors do
    token += Fill_Local_Ledgers(p_ledger[color], p_halo[color], N, D)
  end
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Local Ledgers Done in %.3e s\n", (End-Start)*1e-6)

  -- Initialize Local Ledgers
  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  for color in p_colors do
    token += Consolidate_Local_Ledgers(p_ledger[color], p_local[color], n, color)
  end
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Local Top-n Ledgers Done in %.3e s\n", (End-Start)*1e-6)

  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  var count = Last_Time(r_local, k)
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Final Count Done in %.3e s\n", (End-Start)*1e-6)

  __fence(__execution, __block)
  Start = c.legion_get_current_time_in_micros()
  Update_Global_Ledger(r_global, r_local, k, count)
  __fence(__execution, __block)
  End = c.legion_get_current_time_in_micros()
  c.printf("Global Top-n Ledgers Done in %.3e s\n", (End-Start)*1e-6)

  -- Main Loop
  --var t : double = 0
  --while t < Tf do

    -- Collisions
    --__demand(__index_launch)
    for color in p_particles.colors do
    --  Collision(r_particles[color], p_rng[color], s, Ne, dt, Vc)
    end

    -- Advect
    --__demand(__index_launch)
    for color in p_particles.colors do
    --  Advect(particles[color], p_rng[color], L, w, dt)
    end

    
  --end

  wait_for(token)
  var TS_end = c.legion_get_current_time_in_micros()

  wait_for(token)
  c.printf("Done\n")
  -- Checking values
  c.printf("Checking Values\n")
  for e in r_ledger do
    --c.printf("t[%d] = %f\n", e, r_ledger[e].t)
  end
  for e in r_local do
    c.printf("local_t[%d] = %.3e\n", e, r_local[e].t)
  end
  for e in r_global do
    c.printf("global_t[%d] = %.3e\n", e, r_global[e].t)
  end
  c.printf("Total time: %.6f sec.\n", (TS_end - TS_start) * 1e-6)

  --Dump(r_particles)
end

regentlib.start(toplevel)
