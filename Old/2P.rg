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
  var U : double[2]
  
  c.srand48_r(c.legion_get_current_time_in_nanos(), data)
  c.drand48_r(data, [&double](U))
  c.drand48_r(data, [&double](U))
 
  var g : double
  g = cmath.sqrt(-2*cmath.log(U[0]))*cmath.cos(2*PI*U[1])

  --var g : double[2]
  --g[0] = sqrt(-2*cmath.log(U[0]))*cmath.cos(2*PI*U[1])
  --g[1] = sqrt(-2*cmath.log(U[0]))*cmath.sin(2*PI*U[1])

  return g
end


task initialize(r_particles : region(ispace(int1d), particle),
                r_rng : region(ispace(int1d), c.drand48_data[1]),
                color : int1d, boxes : int32, Ti : double)

where
  reads writes(r_particles),
  reads writes(r_rng)
do
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end

  for e in r_particles do 
    r_particles[e].x = uniform(data)
    r_particles[e].y = uniform(data)
    r_particles[e].z = ([double](color) + uniform(data))/boxes
    --c.printf("Initial position = (%f, %f, %f)\n", r_particles[e].x, r_particles[e].y, r_particles[e].z)
    
    r_particles[e].vx = normal(data)*cmath.sqrt(Ti)
    r_particles[e].vy = normal(data)*cmath.sqrt(Ti)
    r_particles[e].vz = normal(data)*cmath.sqrt(Ti)
  
    r_particles[e].b = color
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
  __fence(__execution, __block)
  c.fclose(g)
end

task Advect(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             Tw : double, Uw : double, dt : double)
where
  reads writes(r_particles.{x,y,z,vx,vy,vz}),
  reads writes(r_rng)
do
  var data : &c.drand48_data
  for e in r_rng do
    data = [&c.drand48_data](@e)
  end

  for p in r_particles do
    --Check if it will hit thermal wall
    var dt1 : double = dt
    var dt2 : double = 0
    if r_particles[p].z + dt*r_particles[p].vz < 0 or r_particles[p].z + dt*r_particles[p].vz > 1 then 
      dt1 = (0-r_particles[p].z)/r_particles[p].vz
      dt2 = dt - dt1
    end
    
    --Advect (either to wall or not)
    r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt1
    r_particles[p].y = r_particles[p].y + r_particles[p].vy*dt1
    r_particles[p].z = r_particles[p].z + r_particles[p].vz*dt1
    
    --If hitting to thermal wall, reset speed and advect
    if dt2 > 0 then
      r_particles[p].vx = cmath.sqrt(Tw)*normal(data)
      r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt2

      r_particles[p].vy = cmath.sqrt(Tw)*normal(data) + Uw
      r_particles[p].y = r_particles[p].y + r_particles[p].vy*dt2

      if r_particles[p].vz < 0 then
        r_particles[p].vz = cmath.sqrt(-2*Tw*cmath.log(uniform(data)))
      elseif r_particles[p].vz > 0 then
        r_particles[p].vz = -r_particles[p].vz
      end
      r_particles[p].z = r_particles[p].z + r_particles[p].vz*dt2
    end
 

    --Periodic BC
    while r_particles[p].x < 0 do
      r_particles[p].x = r_particles[p].x + 1.0
      r_particles[p].y = r_particles[p].y + 1.0
    end
    while r_particles[p].x > 1 do
      r_particles[p].x = r_particles[p].x - 1.0
      r_particles[p].y = r_particles[p].y - 1.0
    end


    
    --while r_particles[p].z < -1 do
    --  r_particles[p].z = r_particles[p].z + 1.0
    --end
    --while r_particles[p].z > 2 do
    --  r_particles[p].z = r_particles[p].z - 1.0
    --end
  
    --Specular BC
    if r_particles[p].z < 0 then
      r_particles[p].z =  0.0 - r_particles[p].z
    end
    if r_particles[p].z > 1 then
      r_particles[p].z =  2.0 - r_particles[p].z
    end
  end
end

    
task Sort(r_particles : region(ispace(int1d), particle), boxes : int32)
where 
  reads (r_particles.z),
  reads writes (r_particles.b)
do
  for p in r_particles do
    r_particles[p].b = [int1d](r_particles[p].z*boxes)
    --c.printf("box = %d, r_particles[p].z = %f\n", r_particles[p].b, r_particles[p].z)
  end
end

--task Rvel(r_particles: region(ispace(int1d), particle),
--          r_v : region(ispace(int1d), vrel))
--where
--  reads (r_particles.{vx,vy,vz}),
--  reads writes (r_v) 
--do
--  for r in r_v do
--  for e in r_particles do
--    -- If initializing, or min/max criteria
--    if r_v[r].xmin == 1234567890 or r_v[r].xmin > r_particles[e].vx then
--      r_v[r].xmin = r_particles[e].vx
--    end
--    if r_v[r].xmax == 1234567890 or r_v[r].xmax < r_particles[e].vx then
--      r_v[r].xmax = r_particles[e].vx
--    end
--    if r_v[r].ymin == 1234567890 or r_v[r].ymin > r_particles[e].vy then
--      r_v[r].ymin = r_particles[e].vy
--    end
--    if r_v[r].ymax == 1234567890 or r_v[r].ymax < r_particles[e].vy then
--      r_v[r].ymax = r_particles[e].vy
--    end
--    if r_v[r].zmin == 1234567890 or r_v[r].zmin > r_particles[e].vz then
--      r_v[r].zmin = r_particles[e].vz
--    end
--    if r_v[r].zmax == 1234567890 or r_v[r].zmax < r_particles[e].vz then
--      r_v[r].zmax = r_particles[e].vz
--    end
--  end
--  end
--  var vmax = 0
--  for r in r_v do
--    vmax += (r_v[r].xmin - r_v[r].xmax)*(r_v[r].xmin - r_v[r].xmax)
--  end
--  return cmath.sqrt(vmax)
--end

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
  reads writes(r_rng)
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
  var vmax : double = Vmax(r_particles)
  var Nc : int32 = r_particles.ispace.volume
  var Ncol : int32 = Nc*Nc*PI*s*s*vmax*Ne*dt/(2*Vc)
  c.printf("Nc = %d, Ncol = %d, vmax = %f\n", Nc, Ncol, vmax)

  --Collide until Ncol
  var ncol : int32 = 0
  while ncol < Ncol do
    --Deviates for selecting particles
    var u1 = int1d(uniform(data)*Nc)
    var u2 = int1d(uniform(data)*Nc)
    var i = lookup_table[u1]
    var j = lookup_table[u2]
    
    --Accept or Reject based on Relative Velocity
    var vrel : double = cmath.sqrt((r_particles[i].vx - r_particles[j].vx)*(r_particles[i].vx - r_particles[j].vx) + (r_particles[i].vy - r_particles[j].vy)*(r_particles[i].vy - r_particles[j].vy) + (r_particles[i].vz - r_particles[j].vz)*(r_particles[i].vz - r_particles[j].vz)) 
    if vrel/vmax > uniform(data) then
      
      --Update Counter
      ncol += 1
      
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
  return 1/(cmath.sqrt(2)*PI*s*s*n)
end

terra MFT(s : double, n : double, T : double)
  return MFP(s,n)/cmath.sqrt(8/PI*T)
end

task toplevel()
  var config : Config
  config:initialize_from_command()

  -- Simulation Parameters
  var n : double = 1e-3          -- Initial Number Density
  var s : double = 1.0           -- CrossSection Sigma 
  var l : double = MFP(s,n)      -- Mean Free Path
  var boxes : int32 = 50         -- Number of Boxes
  var L : double = l*0.2*boxes   -- Height of Box
  var w : double = l*0.2         -- Width of One Cell (Cube)
  var N : int32 = 5000           -- Number of MC Particles
  var Ne : double = N/L/w/w/n    -- Effective Particles per MC Particle
  var Ti : double = 1.0          -- Initial Temperature of Gas -- NEED TO MAKE IT DO THE THING
  var Tw : double = 1.0          -- Wall Temperature
  var Uw : double = 1.0          -- Wall Velocity
  var Vc : double = w*w*w        -- Volume of Cell
  var t  : double = MFT(s,n,Ti)  -- Mean Free Time
  var dt : double = t/25         -- Timestep
  var maxiter : int32 = 5        -- Timesteps

  c.printf("%f Effective Number of Particles Per MC Particle\n", Ne)

  -- Create a logical region for particles
  var r_particles = region(ispace(int1d, N), particle)
 
  -- Create an equal partition of the particles
  var p_colors = ispace(int1d, boxes)
  var p_particles = partition(equal, r_particles, p_colors)

  -- Create a region for random number generators
  var r_rng = region(p_colors, c.drand48_data[1])
  var p_rng = partition(equal, r_rng, p_colors)

  -- Create a region to collect relative velocities
  --var r_v = region(p_colors, vrel)
  --var p_v = partition(equal, r_v, p_colors)

  -- Initialization
  var token : int32 = 0
  for color in p_colors do
    initialize(p_particles[color],p_rng[color], color, boxes, Ti)
    token += block_task(p_particles[color])
  end
  wait_for(token)

  var iter : int32 = 0
  var TS_start = c.legion_get_current_time_in_micros()  
  for iter = 0, maxiter do
    for color in p_colors do
      initialize(p_particles[color],p_rng[color], color, boxes, Ti)
      token += block_task(p_particles[color])
    end
    wait_for(token)
    c.printf("Current Iteration : %d\n", iter)    
    -- Repartition based on Color Field
    var particles = partition(r_particles.b, p_colors)  
    
    -- Collisions
    for color in particles.colors do
      Collision(particles[color], p_rng[color], s, Ne, dt, Vc)
    end

    -- Advect
    for color in particles.colors do
      Advect(particles[color], p_rng[color], Tw, Uw, dt)
    end

    -- Sort into Boxes
    for color in particles.colors do
      Sort(particles[color], boxes)
    end

    Dump(r_particles, iter)
  end --this one ends the 'while T < endtime' loop
  
  for color in p_colors do
    token += block_task(p_particles[color])
  end
  wait_for(token)


  var TS_end = c.legion_get_current_time_in_micros()

  c.printf("Done\n")
  c.printf("Total time: %.6f sec.\n", (TS_end - TS_start) * 1e-6)

  --Dump(r_particles)
end

regentlib.start(toplevel)
