import "regent"

-- Helper modules to handle PNG files and command line arguments
local Config = require("config")
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
   --m : double,   
}

-- Field space for collision time ledgers
fspace ledger
{
  t : double,
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
                N : int64, D : double)
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
  var sod : Sod
  sod:initialize_parameters()
  var NL : int64 = N*(sod.pL/(sod.pL + sod.pR))
  var NR : int64 = N*(sod.pR/(sod.pL + sod.pR))
  if (NL + NR) < N then
    NR += 1
  end

  for e in r_particles do
    if e < NL then
      r_particles[e].x = uniform(data)*(0.5/NL - D) + 0.5*[double](e)/double](NL) + D/2
      r_particles[e].vx = normal(data)*cmath.sqrt(2*sod.PL/sod.pL)
    else
      r_particles[e].x  =  uniform(data)*(0.5/NR - D) + 0.5*[double](e)/double](NR) + 0.5 + D/2
      r_particles[e].vx = normal(data)*cmath.sqrt(2*sod.PR/sod.pR)
    end
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

    if l == -1 then 
      var lx : double = -D/2
      var lv : double = 0
    else
      var lx : double = r_particles[l].x
      var lv : double = r_particles[l].vx
    end

    if r == N then
      var rx : double = 1.0 + D/2
      var rv : double = 0
    else
      var rx : double = r_particles[r].x
      var rv : double = r_particles[r].vx
    end

    r_ledger[e].t = (rx - lx - D)/(lv - rv)
  end
end

task Consolidate_Local_Ledgers(r_ledger : region(ispace(int1d), ledger),
                r_top : region(ispace(int1d), ledger),
                N : int64, D : double, K : int32)
where 
  reads writes (r_top),
  reads (r_ledger)
do
  var k : int32 = 0
  var top : double[K]
  for e in r_ledger do
    var continue : bool = true
    
    while continue do

      -- If current time is shorter than top[k], stop while, push back all times, and set top[k] to current time
      if r_ledger[e].t < top[k] then
        -- Stop while
        continue = false

        -- Pushback Times
        for j = k, K-1 do
          var top_k : double = top[j+1]
          top[j+1] = top[j]
        end

        -- Update
        top[k] = r_ledger[e].t
      end

      -- Otherwise, continue
      k += 1
    end
  end
end

task Update_Global_Ledger(global : region(ispace(int1d), ledger),
                          local : region(ispace(int1d), ledger),
                N : int64, D : double)
where 
  reads writes 


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
    dumpdouble(g, r_grid[e].z)
  end
  __fence(__execution, __block)
  for e in r_grid do
    dumpdouble(g, r_grid[e].vy)
  end
  for e in r_grid do
    dumpdouble(g, r_grid[e].vx)
  end
  for e in r_grid do
    dumpdouble(g, r_grid[e].vz)
  end
  __fence(__execution, __block)
  c.fclose(g)

  return 1
end

task Advect(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             L : double, w : double, dt : double)
where
  reads writes(r_particles.{x,y,z,vx,vy,vz}),
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

      if r_particles[p].z + dt1*r_particles[p].vz < 0 then 
        dt1 = (0 - r_particles[p].z)/r_particles[p].vz
      elseif r_particles[p].z + dt1*r_particles[p].vz > L then
        dt1 = (L - r_particles[p].z)/r_particles[p].vz
      end        
    
      --Advect (either to wall or not)
      r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt1
      r_particles[p].y = r_particles[p].y + r_particles[p].vy*dt1
      r_particles[p].z = r_particles[p].z + r_particles[p].vz*dt1

      --Periodic BC
      --while r_particles[p].x < 0 do
      --  r_particles[p].x = r_particles[p].x + 1.0*w
      --end
      --while r_particles[p].y < 0 do
      --  r_particles[p].y = r_particles[p].y + 1.0*w
      --end
      --while r_particles[p].x > w do
      --  r_particles[p].x = r_particles[p].x - 1.0*w
      --end
      --while r_particles[p].y > w do
      --  r_particles[p].y = r_particles[p].y - 1.0*w
      --end

      --Periodic BC
      if r_particles[p].x < 0 or r_particles[p].x > w then
        r_particles[p].x = cmath.fmod(r_particles[p].x, w)
      end
      if r_particles[p].y < 0 or r_particles[p].y > w then
        r_particles[p].y = cmath.fmod(r_particles[p].y, w)
      end
      --Reflective BC

      if dt1 < dt-tsim then
        r_particles[p].vz = -r_particles[p].vz
      end
      --Update Tsim Counter
      tsim += dt1
    end
  end
end

    
task Sort(r_particles : region(ispace(int1d), particle), boxes : int32, L : double)
where 
  reads (r_particles.z),
  reads writes (r_particles.b)
do
  for p in r_particles do
    r_particles[p].b = [int1d](r_particles[p].z/L*boxes)
    --c.printf("box = %d, r_particles[p].z = %f\n", r_particles[p].b, r_particles[p].z)
  end
end


task Collision(r_particles : region(ispace(int1d), particle),
               r_rng : region(ispace(int1d), c.drand48_data[1]),
               s: double, Ne : double, dt : double, Vc : double)
where
  reads writes(r_particles.{vx,vy,vz}),
  reads writes(r_rng)--,
  --exclusive(r_particles.{x,y,z,vx,vy,vz})
do
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end

  --Define Makeshift Hash Table
  var lookup_table = region(ispace(int1d, r_particles.ispace.volume), int1d)
  var next_id = 0
  for i in r_particles do
    lookup_table[next_id] = i
    next_id += 1
  end

  --Compute Number of Collisions Required
  var xmin : double = 1234567890
  var xmax : double = 1234567890
  var ymin : double = 1234567890
  var ymax : double = 1234567890
  var zmin : double = 1234567890
  var zmax : double = 1234567890

  for e in r_particles do
    -- If initializing, or min/max criteria
    if xmin == 1234567890 or xmin > r_particles[e].vx then
      xmin = r_particles[e].vx
    end
    if xmax == 1234567890 or xmax < r_particles[e].vx then
      xmax = r_particles[e].vx
    end
    if ymin == 1234567890 or ymin > r_particles[e].vy then
      ymin = r_particles[e].vy
    end
    if ymax == 1234567890 or ymax < r_particles[e].vy then
      ymax = r_particles[e].vy
    end
    if zmin == 1234567890 or zmin > r_particles[e].vz then
      zmin = r_particles[e].vz
    end
    if zmax == 1234567890 or zmax < r_particles[e].vz then
      zmax = r_particles[e].vz
    end
  end

  var vmax : double = cmath.sqrt((xmax-xmin)*(xmax-xmin) + (ymax-ymin)*(ymax-ymin) + (zmax-zmin)*(zmax-zmin))

  --var vmax : double = Vmax(r_particles)
  var Nc : int32 = r_particles.ispace.volume
  var Ncand : int64 = Nc*Nc*PI*s*s*vmax*Ne*dt/(2*Vc)
  --c.printf("Nc = %d, Ncand = %d, vmax = %f, Ne = %f, dt = %f, Vc = %.30f\n", Nc, Ncand, vmax, Ne, dt, Vc)
  if Ncand == 0 then
    --c.printf("Number of Collision Candidates is zero!\n") 
  end 

  --Collide until Ncand
  var ncol : int32 = 0
  var ncand : int32 = 0
  while ncand < Ncand do
    --Update Counter
    ncand += 1
    
    --c.printf("Candidate Total: %d, Candidates Selected:%d \n", Ncand, ncand)      

    --Deviates for selecting particles
    var u1 = int1d(uniform(data)*Nc)
    var u2 = int1d(uniform(data)*Nc)
    var i = lookup_table[u1]
    var j = lookup_table[u2]
    
    --Accept or Reject based on Relative Velocity
    var vrel : double = cmath.sqrt((r_particles[i].vx - r_particles[j].vx)*(r_particles[i].vx - r_particles[j].vx) + (r_particles[i].vy - r_particles[j].vy)*(r_particles[i].vy - r_particles[j].vy) + (r_particles[i].vz - r_particles[j].vz)*(r_particles[i].vz - r_particles[j].vz)) 
    if vrel/vmax > uniform(data) then
      

      --Deviates for selecting angles
      var a : double = 2*PI*uniform(data)  -- phi
      var b : double = 2*uniform(data) - 1 -- cos(theta)
    
    
      --Momentum Exchange
      r_particles[i].vx = (r_particles[i].vx + r_particles[j].vx)/2 + vrel*cmath.sqrt(1-b*b)*cmath.cos(a)/2
      r_particles[j].vx = (r_particles[i].vx + r_particles[j].vx)/2 - vrel*cmath.sqrt(1-b*b)*cmath.cos(a)/2

      r_particles[i].vy = (r_particles[i].vy + r_particles[j].vy)/2 + vrel*cmath.sqrt(1-b*b)*cmath.sin(a)/2
      r_particles[j].vy = (r_particles[i].vy + r_particles[j].vy)/2 - vrel*cmath.sqrt(1-b*b)*cmath.sin(a)/2

      r_particles[i].vz = (r_particles[i].vz + r_particles[j].vz)/2 + vrel*b/2
      r_particles[j].vz = (r_particles[i].vz + r_particles[j].vz)/2 - vrel*b/2
    end
  end
  __fence(__execution, __block)
  --c.printf("Collision is Done\n")
end

terra MFP(s : double, n : double)
  return 1.0/(cmath.sqrt(2)*PI*s*s*n)
end

terra MFT(s : double, n : double, T : double, L: double)
  return MFP(s,n)/cmath.sqrt(8/PI*T)/L
end


task toplevel()
  var config : Config
  config:initialize_from_command()

  -- Simulation Parameters
  var tf : double = 0.25           -- Final Time
  var dt : double = t/2            -- Initial Timestep
  var out : bool = config.out      -- Output Boolean

  --c.printf("N = %d, %.5f\n", N, Ne)

  -- Create a logical region for particles and ledgers
  var r_particles = region(ispace(int1d, N), particle)
  var r_ledger = region(ispace(int1d, boxes), box) 

  -- Create an equal partition of the particles
  var p_colors = ispace(int1d, boxes)
  var p_particles = partition(equal, r_particles, p_colors)
  var p_box = partition(equal, r_box, p_colors)

  -- Create a region for random number generators
  var r_rng = region(p_colors, c.drand48_data[1])
  var p_rng = partition(equal, r_rng, p_colors)

  var token : int32 = 0
  var TS_start = c.legion_get_current_time_in_micros()   
  __forbid(__parallel)
  var sim_start = c.legion_get_current_time_in_micros()
   
  -- Initialization 
  if rel == start then
    __demand(__parallel)
    for color in p_colors do
      token += Initialize(p_particles[color],p_rng[color], color, boxes, TL, TR, pL, pR, L, w)
    end
  end

  -- Iterate for maxiter timesteps
  var iter : int32 = 0
  __forbid(__parallel)
  for iter = 1, maxiter+1 do
    __fence(__execution, __block)
    c.printf("Current Realization : %d, Current Iteration : %d\n", rel, iter)    
    -- Repartition based on Color Field
    var particles = partition(r_particles.b, p_colors)  
  
    -- Collisions
    __demand(__parallel)
    for color in particles.colors do
      Collision(particles[color], p_rng[color], s, Ne, dt, Vc)
    end

    -- Advect
    __demand(__parallel)
    for color in particles.colors do
      Advect(particles[color], p_rng[color], L, w, dt)
    end

    -- Sort into Boxes
    __demand(__parallel)
    for color in particles.colors do
      Sort(particles[color], boxes, L)
    end

    if (iter%(5)) == 0 and out then
      grid += 1
      token += Dump(r_particles, grid)
    end
    if (iter%(5)) == 0 and out then
      wait_for(token)
      c.printf("Dumped Grid %d\n", grid)
    end

    if iter == maxiter and rel < reals then
      __demand(__parallel)
      for color in p_colors do
        token += initialize(p_particles[color],p_rng[color], color, boxes, TL, TR, pL, pR, L, w)
      end
    end
  
    wait_for(token)
    var sim_end = c.legion_get_current_time_in_micros()
    c.printf("Simulation took %.6f sec\n", (sim_end - sim_start)*1e-6) 
  end

  var TS_end = c.legion_get_current_time_in_micros()

  wait_for(token)
  c.printf("Done\n")
  c.printf("Total time: %.6f sec.\n", (TS_end - TS_start) * 1e-6)

  --Dump(r_particles)
end

regentlib.start(toplevel)