import "regent"

-- Helper modules to handle PNG files and command line arguments
local EdgeConfig = require("edge_config")
local coloring   = require("coloring_util")
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
   b : ptr,
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
                r_rng : region(ispace(int1d), c.drand48_data[1]))
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
    r_particles[e].z = uniform(data)

    r_particles[e].vx = normal(data)
    r_particles[e].vy = normal(data)
    r_particles[e].vz = normal(data)
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


task Dump(r_particles : region(ispace(int1d), particle))
where 
  reads (r_particles)
do
  var f = c.fopen('particles','wb')
    
  for e in r_particles do
    dumpint32(f,e.x)
  end
  
  for e in r_particles do
    dumpint32(f,e.y)
  end

  for e in r_particles do
    dumpint32(f,e.z)
  end    
end




task advectx(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             Tw : double, dt : double)
where
  reads (r_particles.{vx,vz,z,x}),
  writes (r_particles.{vx,x}),
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
    if r_particles[p].z + dt*r_particles[p].vz < 0 then 
      dt1 = (0-r_particles[p].z)/r_particles[p].vz
      dt2 = dt - dt1
    end
    
    --Advect (either to thermal wall or not)
    r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt1
    
    --If hitting to thermal wall, reset speed and advect
    if dt2 > 0 then
      r_particles[p].vx = cmath.sqrt(Tw)*normal(data)
      r_particles[p].x = r_particles[p].x + r_particles[p].vx*dt2
    end
 

    --Periodic BC
    while r_particles[p].x < 0 do
      r_particles[p].x = r_particles[p].x + 1.0
    end
    while r_particles[p].x > 1 do
      r_particles[p].x = r_particles[p].x - 1.0
    end
  
  end
end


task advecty(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             Tw : double, Uw : double, dt : double)
where
  reads (r_particles.{vy,vz,z,y}),
  writes (r_particles.{vy,y}),
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

    if r_particles[p].z + dt*r_particles[p].vz < 0 then 
      dt1 = (0-r_particles[p].z)/r_particles[p].vz
      dt2 = dt - dt1
    end
    
    --Advect (either to thermal wall or not)
    r_particles[p].y = r_particles[p].y + r_particles[p].vy*dt1
    
    --If hitting to thermal wall, reset speed and advect
    if dt2 > 0 then
      r_particles[p].vy = cmath.sqrt(Tw)*normal(data) + Uw
      r_particles[p].y = r_particles[p].y + r_particles[p].vy*dt2
    end
 

    --Periodic BC
    while r_particles[p].y < 0 do
      r_particles[p].y = r_particles[p].y + 1.0
    end
    while r_particles[p].y > 1 do
      r_particles[p].y = r_particles[p].y - 1.0
    end
  
  end
end

task advectz(r_particles : region(ispace(int1d), particle),
             r_rng : region(ispace(int1d), c.drand48_data[1]),
             Tw : double, dt : double)
where
  reads (r_particles.{vz,z}),
  writes (r_particles.{vz,z}),
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

    if r_particles[p].z + dt*r_particles[p].vz < 0 then 
      dt1 = (0-r_particles[p].z)/r_particles[p].vz
      dt2 = dt - dt1
    end

    
    --Advect (either to thermal wall or not)
    r_particles[p].z = r_particles[p].z + r_particles[p].vz*dt1
    
    --If hitting thermal wall, reset speed and advect
    if dt2 > 0 then
      r_particles[p].vz = cmath.sqrt(-2*Tw*cmath.log(uniform(data)))
      r_particles[p].z = r_particles[p].z + r_particles[p].vz*dt2
    end
 

    --If anomalous velocity takes it far, bring it 1 "periodic cell" away before reflecting
    while r_particles[p].z < -1 do
      r_particles[p].z = r_particles[p].z + 1.0
    end
    while r_particles[p].z > 2 do
      r_particles[p].z = r_particles[p].z - 1.0
    end
  
    if r_particles[p].z < 0 then
      r_particles[p].z = -1.0 * r_particles[p].z
    end
    if r_particles[p].z > 1 then
      r_particles[p].z =  2.0 - r_particles[p].z
    end
  end
end

task Sort(r_particles : region(ispace(int1d), particle), boxes : int32)
where 
  reads (r_particles.z),
  writes (r_particles.b)
do
  var dz : double = 1.0/boxes
  for p in r_particles do
    r_particles[p].b = ptr(1.0/dz*r_particles[p].z)
  end
end

task Collisionx(r_particles : region(ispace(int1d), particle),
                r_rng : region(ispace(int1d), c.drand48_data[1]))
where
  reads writes(r_particles.
do
  var lookup_table = region(ispace(int1d, r_particles.ispace.volume), int1d)
  var next_id = 0
  for i in r_particles do
    lookup_table[next_id] = i
    next_id += 1
  end


end

task toplevel()
  var config : EdgeConfig
  config:initialize_from_command()

  -- Create a logical region for particles
  var N : int32 = 50000
  var Ne : int32 = 50
  var boxes : int32 = 50

  var r_particles = region(ispace(int1d, N), particle)
 
  -- Create an equal partition of the particles
  var p_colors = ispace(int1d, boxes)
  var p_particles = partition(equal, r_particles, p_colors)


  var r_rng = region(p_colors, c.drand48_data[1])
  var p_rng = partition(equal, r_rng, p_colors)

  --Initialization
  var token : int32 = 0
  for color in p_colors do
    initialize(p_particles[color],p_rng[color])
    token += block_task(p_particles[color])
  end
  wait_for(token)

  var iter : int32 = 0
  var maxiter : int32 = 1
  var Tw : double = 1.0
  var Uw : double = 0.0
  var dt : double = 0.01
  var TS_start = c.legion_get_current_time_in_micros()  
  while iter < maxiter do
    iter += 1
    for color in p_particles.colors do
      Sort(p_particles[color], boxes)
    end

    for color in p_particles.colors do
      advectx(p_particles[color], p_rng[color], Tw, dt)
    end
    for color in p_particles.colors do
      advecty(p_particles[color], p_rng[color], Tw, Uw, dt)
    end
    for color in p_particles.colors do
      advectz(p_particles[color], p_rng[color], Tw, dt)
    end
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
