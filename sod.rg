import "regent"

local c = regentlib.c

struct Sod
{
  pL : double,
  pR : double,
  PL : double,
  PR : double,
  vL : double,
  vR : double
}


local cstring = terralib.includec("string.h")


terra Sod:initialize_parameters()

  self.pL = 1.0
  self.pR = 1.0/8.0
  self.PL = 1.0
  self.PR = 1.0/10.0
  self.vL = 0
  self.vR = 0

end

return Sod
