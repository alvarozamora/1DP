import "regent"

-- Helper modules to handle PNG files and command line arguments
local Config = require("config")
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
   y  : double,
   z  : double,
   vx : double,
   vy : double,
   vz : double,
   b : int1d,
}

fspace box
{
  n : int32
}

fspace vrel
{
  xmax : double,
  xmin : double,
  ymax : double,
  ymin : double,
  zmax : double,
  zmin : double
}

terra uniform(data : &c.drand48_data)
  var flip : double[1]
  c.srand48_r(c.legion_get_current_time_in_nanos(), data)
  c.drand48_r(data, [&double](flip))
  return flip[0]
end

terra normal(data : &c.drand48_data)
  var U1 : double[1]
  var U2 : double[1]
  
  c.srand48_r(c.legion_get_current_time_in_nanos(), data)
  c.drand48_r(data, [&double](U1))
  c.drand48_r(data, [&double](U2))
 
  var g : double
  g = cmath.sqrt(-2*cmath.log(U1[0]))*cmath.cos(2*PI*U2[0])

  --var g : double[2]
  --g[0] = sqrt(-2*cmath.log(U[0]))*cmath.cos(2*PI*U[1])
  --g[1] = sqrt(-2*cmath.log(U[0]))*cmath.sin(2*PI*U[1])

  return g
end


task initialize(r_particles : region(ispace(int1d), particle),
                r_rng : region(ispace(int1d), c.drand48_data[1]),
                color : int1d, boxes : int32, TL : double, TR : double, pL : double, pR : double, L : double, w : double)
where
  reads writes(r_particles),
  reads writes(r_rng)
do
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end

  for e in r_particles do 
    if uniform(data) < pL/(pL+pR) then 
      r_particles[e].x = uniform(data)*w
      r_particles[e].y = uniform(data)*w
      r_particles[e].z = uniform(data)*L/2
    else
      r_particles[e].x = uniform(data)*w
      r_particles[e].y = uniform(data)*w
      r_particles[e].z = (uniform(data) + 1)*L/2
    end

 
    --c.printf("Initial position = (%f, %f, %f)\n", r_particles[e].x, r_particles[e].y, r_particles[e].z)
    
    if r_particles[e].z < 0.5 then
      r_particles[e].vx = normal(data)*cmath.sqrt(TL)
      r_particles[e].vy = normal(data)*cmath.sqrt(TL)
      r_particles[e].vz = normal(data)*cmath.sqrt(TL)
    else
      r_particles[e].vx = normal(data)*cmath.sqrt(TR)
      r_particles[e].vy = normal(data)*cmath.sqrt(TR)
      r_particles[e].vz = normal(data)*cmath.sqrt(TR)
    end
    --c.printf("Initial velocity = (%f, %f, %f)\n", r_particles[e].vx, r_particles[e].vy, r_particles[e].vz)
  
    r_particles[e].b = int1d(boxes*r_particles[e].z/L)
  end
  return 1
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

task factorize2d(parallelism : int) : int2d
  var limit = [int](cmath.sqrt([double](parallelism)))
  var size_x = 1
  var size_y = parallelism
  for i = 1, limit + 1 do
    if parallelism % i == 0 then
      size_x, size_y = i, parallelism / i
      if size_x > size_y then
        size_x, size_y = size_y, size_x
      end
    end
  end
  return int2d { size_x, size_y }
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
      while r_particles[p].x < 0 do
        r_particles[p].x = r_particles[p].x + 1.0*w
        r_particles[p].y = r_particles[p].y + 1.0*w
      end
      while r_particles[p].x > L do
        r_particles[p].x = r_particles[p].x - 1.0*w
        r_particles[p].y = r_particles[p].y - 1.0*w
      end
    
      --If hit thermal wall, reset speed
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

task Vmax(r_particles: region(ispace(int1d), particle))
where
  reads (r_particles.{vx,vy,vz})
do
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

  return cmath.sqrt((xmax-xmin)*(xmax-xmin) + (ymax-ymin)*(ymax-ymin) + (zmax-zmin)*(zmax-zmin))
end


task Collision(r_particles : region(ispace(int1d), particle),
               r_rng : region(ispace(int1d), c.drand48_data[1]),
               s: double, Ne : double, dt : double, Vc : double)
where
  reads writes(r_particles.{x,y,z,vx,vy,vz}),
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
  var Ncand : int32 = Nc*Nc*PI*s*s*vmax*Ne*dt/(2*Vc)
  --c.printf("Nc = %d, Ncand = %d, vmax = %f\n", Nc, Ncand, vmax)
  if Ncand == 0 then
    --c.printf("Number of Collision Candidates is zero!\n") 
  end 

  --Collide until Ncand
  var ncol : int32 = 0
  var ncand : int32 = 0
  while ncand < Ncand do
    --Update Counter
    ncand += 1
    --c.printf("Collision Count: %d, Candidates Selected:%d \n", ncol, ncand)      

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
end

terra MFP(s : double, n : double)
  return 1.0/(cmath.sqrt(2)*PI*s*s*n)
end

terra MFT(s : double, n : double, T : double)
  return MFP(s,n)/cmath.sqrt(8/PI*T)
end

task Rayleigh(r_particles : region(ispace(int1d), particle), boxes : int32, L : double)
where
  reads (r_particles.{z,vy})
do
  var wallspeed : double = 0 
  var npart : int32 = 0
 
  for e in r_particles do
    if r_particles[e].z < L/boxes then
      wallspeed += r_particles[e].vy
      npart += 1
    end
  end
  return wallspeed/npart
end

task Testing(r_particles : region(ispace(int1d), particle), boxes : int32, N : int32)
where
  reads (r_particles.{z,vy,vz,vx})
do
  -- First Moment
  var x1 : double = 0
  var y1 : double = 0
  var z1 : double = 0

  for p in r_particles do
    x1 += r_particles[p].vz
     --unfinished
  end 

end

task toplevel()
  var config : Config
  config:initialize_from_command()

  -- Simulation Parameters
  var n : double = 1e6           -- Initial Number Density
  var s : double = 1.0           -- CrossSection Sigma 
  var l : double = MFP(s,n)      -- Mean Free Path
  var boxes : int32 = 50        -- Number of Boxes
  var L : double = 10*l          -- Height of Box
  var w : double = 1e4/cmath.sqrt(n*l)  -- X/Y Width of One Cell
  var N : int32 = 1e5            -- Number of MC Particles
  var Ne : double = n/(N/L/w/w)  -- Effective Particles per MC Particle
  var TL : double = 1.0          -- Initial Temperature of Gas
  var TR : double = 8.0/10.0     -- Initial Temperature of Gas
  var pL : double = 1.0          -- Initial Density of Gas
  var pR : double = 1.0/8.0      -- Initial Density of Gas
  var Vc : double = L*w*w/boxes  -- Volume of Cell
  var t  : double = MFT(s,n,TL)  -- MFT 
  var dt : double = 0*t/1000       -- Timestep
  var maxiter : int32 = 1       -- Timesteps
  var reals : int32 = config.last  -- Realizations
  var start : int32 = config.start -- Resuming Point
  var out : bool = config.out      -- Output Boolean

  c.printf("N = %d, %.15f Effective Number of Particles Per MC Particle\n", N, Ne)
  c.printf("The Mean Free Path is %.15f, and the Mean Free Time is %.15f\n", l, t)
  c.printf("Boxes = %d\n", boxes)
  c.printf("Box Size = (%f, %f, %f)", w, w, L)

  var grid : int32 = start

  -- Create a logical region for particles
  var r_particles = region(ispace(int1d, N), particle)
  var r_box = region(ispace(int1d, boxes), box) 

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
  for rel = start, reals do
    wait_for(token)
    var rel_start = c.legion_get_current_time_in_micros()
   
    -- Initialization 
    if rel == start then
      __demand(__parallel)
      for color in p_colors do
        token += initialize(p_particles[color],p_rng[color], color, boxes, TL, TR, pL, pR, L, w)
      end
    end

    -- Iterate for maxiter timesteps
    var iter : int32 = 0
    __forbid(__parallel)
    for iter = 1, maxiter+1 do
      wait_for(token)
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

      if (iter%(1)) == 0 and out then
        grid += 1
        token += Dump(r_particles, grid)
      end
      if (iter%(1)) == 0 and out then
        wait_for(token)
        c.printf("Dumped Grid %d\n", grid)
      end

      if iter == maxiter and rel < reals then
        __demand(__parallel)
        for color in p_colors do
          token += initialize(p_particles[color],p_rng[color], color, boxes, TL, TR, pL, pR, L, w)
        end
      end
    end --this one ends the 'while T < endtime' loop

  
    wait_for(token)
    var rel_end = c.legion_get_current_time_in_micros()
    c.printf("Realization %d took %.6f sec\n", rel, (rel_end-rel_start)*1e-6) 
  end

  var TS_end = c.legion_get_current_time_in_micros()

  wait_for(token)
  c.printf("Done\n")
  c.printf("Total time: %.6f sec.\n", (TS_end - TS_start) * 1e-6)

  --Dump(r_particles)
end

regentlib.start(toplevel)
