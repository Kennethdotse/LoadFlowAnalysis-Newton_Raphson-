import numpy as np
import sympy as sp
import bisect




#Class for the various quantites of interest for Load flow studies [P, Q, V, D]
class Qty:
  def __init__(self, type, bus, value ):
    self.bus = bus
    self.type = type
    self.value = value
    
  @property
  def type(self):
    return self._type
  
  @type.setter
  def type(self, val):
    types = ['P', 'Q', 'D', 'V']
    if not val:
      raise ValueError("type cannot be empty")
    elif val.upper() not in types:
      raise ValueError("Invalid Type. Type must be: [P, Q, D, V]")
    self._type = val.upper()
  
  @property
  def bus(self):
    return self._bus
  
  @bus.setter
  def bus(self, val):
    if not val:
      raise ValueError("bus cannot be empty")
    elif not isinstance(val, int) or val < 0:
      raise ValueError("Value must be a positive Integer")
    self._bus = val
    
  @property 
  def value(self):
    return self._value
  
  @value.setter
  def value(self, val):
    self._value = val
    
    
    



# Class for the Vector of Known Quantities
class Kvector:
  def __init__(self):
    self.data = []
    
  #adds quantities objects to the vector but ensures they are only of type P and Q and arranges them in the right order
  def push(self, qty):
    types = ['P', 'Q']
    if not isinstance(qty, Qty):
      raise TypeError(f'Expected a Qty object, but got {type(qty).__name__} instead.')
    if qty.type.upper() not in types:
      raise TypeError("cannot add V or D quantities to Vector of Knowns")
    
    # Rest of the code below is for sorting Vector so that P's come before Q's and they are arranged in ascending order of bus numbers
    
    type_order = {'P': 0, 'Q': 1}
    new_entry = (type_order[qty.type], qty.bus)
        
    # Using bisect to find the correct insertion index
    sortable_entries = [(type_order[obj.type], obj.bus) for obj in self.data]
    index = bisect.bisect_left(sortable_entries, new_entry)
    self.data.insert(index, qty)
    
    
  #for displaying the array in a user-friendly way in the console when you print objects of the class
  def __repr__(self):
    return repr([f"{qty.type}{qty.bus}: {qty.value}" for qty in self.data])
    
    


    
    
# Class for the Vector of Unknown Quantities
class Uvector:
  def __init__(self):
    self.data = []
    
  #adds quantities to the unknown vector but ensures they are only of type V and D and arranges them in the right order
  def push(self, qty):
    types = ['V', 'D']
    if not isinstance(qty, Qty):
      raise TypeError(f'Expected a Qty object, but got {type(qty).__name__} instead.')
    if qty.type.upper() not in types:
      raise TypeError("cannot add P or Q quantities to Vector of Unknowns")
    
    # For sorting the Vector so that D's come before V's and they are arranged in ascending order of bus numbers
    type_order = {'D': 0, 'V': 1}
    new_entry = (type_order[qty.type], qty.bus)
        
    # Using bisect to find the correct insertion index
    sortable_entries = [(type_order[obj.type], obj.bus) for obj in self.data]
    index = bisect.bisect_left(sortable_entries, new_entry)
    self.data.insert(index, qty)
    
    
  #for displaying the array in a user-friendly way in the console
  def __repr__(self):
    return repr([f"{qty.type}{qty.bus}: {qty.value}" for qty in self.data])
    



 
#Function to calculate P. Takes arguments of the bus, delta vector, Voltage vector and Addmittance matrix
def calc_P_i(bus, D, V, Y):
  i = bus-1  # Index for P_i
  
  #Checks arrays to ensure they are of matching order
  if D.shape != V.shape:
    raise TypeError("V matrix and D matrix must have the same order")
  elif len(Y) != len(V) and len(Y[0] != len(V)):
    raise TypeError(f"Admittance matrix must be a {len(V)} by {len(V)} matrix")

  sum_term = 0
  for j in range(len(V)):
    #converting angles from delta array into radians
    # Dj_rad = np.radians(D[j])
    # Di_rad = np.radians(D[i])
    Dj_rad = D[j]
    Di_rad = D[i]
    
    #computing summation terms first
    cos_term = np.cos(np.angle(Y[i, j]) + Di_rad - Dj_rad)   #this works even though it should be Dj_rad - Di_rad
    sum_term += V[j] * np.abs(Y[i, j]) * cos_term

  P_i = V[i] * sum_term   #multiplying summation term with V

  # print(f"P[{bus}]:", P_i)
  
  return P_i
  



  
  
#Function to calculate Q. Takes the same arguments as for P above
def calc_Q_i(bus, D, V, Y):
  i = bus-1  # Index for Q_i
  
  #Checks arrays to ensure they are of matching order
  if D.shape != V.shape:
    raise TypeError("V matrix and D matrix must have the same order")
  elif len(Y) != len(V) and len(Y[0] != len(V)):
    raise TypeError(f"Admittance matrix must be a {len(V)} by {len(V)} matrix")

  sum_term = 0
  for j in range(len(V)):
    #converting angles from delta array into radians
    # Dj_rad = np.radians(D[j])
    # Di_rad = np.radians(D[i])
    
    Dj_rad = D[j]
    Di_rad = D[i]
    
    #computing summation terms first
    cos_term = np.sin(np.angle(Y[i, j]) + Di_rad - Dj_rad)   #this works even though it should be Dj_rad - Di_rad
    sum_term += V[j] * np.abs(Y[i, j]) * cos_term

  Q_i = -V[i] * sum_term   #multiplying summation term with V

  # print(f"Q[{bus}]:", Q_i)
  
  return Q_i





#Function for calculating the Mismatch vector. 
# Takes in arguments: 
# Vector of specified values:
# delta vector for angles of V
# Voltage vector
# Admittance Matrix

# NOTE!!!: The vector of specified values MUST not be a normal array. It should be an
#instance of the Kvector class 'filled' with objects of the specified values

def MismatchV(spec, D, V, Y):
  # Initialized empty Vector for calculated Values of P and Q's 
  calc = Kvector()

  #Calculating for Quantities corresponding to the same quantities in the specified vector
  for qty in spec.data:
    if qty.type == "P":
      val = calc_P_i(qty.bus, D, V, Y)
    elif qty.type == "Q":
      val =calc_Q_i(qty.bus, D, V, Y)
      
    entry = Qty(qty.type, qty.bus, val)
    calc.push(entry) 

  #NOTE!! This function returns the kVector object, mismatch, initialized below  
  mismatch = Kvector()

  #Calculating differences between specified data and calculated data
  order = len(calc.data)    
  for i in range(order):
    qty_cal = calc.data[i]
    qty_spec = spec.data[i]
    calc_val = qty_spec.value - qty_cal.value
    entry = Qty(qty_cal.type, qty_cal.bus, calc_val)
    mismatch.push(entry)
    
  
    
  return mismatch
  



def Jacobian(Kvector, Uvector, D, V, Y):
  JacobV = np.empty(0)
  for qty in Kvector.data:
    i = qty.bus
    
    #Generating the buses numbers for variables in the equation of each Known Value
    j_indices = [idx + 1 for idx in range(len(Kvector.data)) ]
  
    
    # Dynamically defining variables of corresponding buses
    Vi = sp.symbols(f"V{i}")
    Vj = [sp.symbols(f"V{j}") for j in j_indices]
    Di = sp.symbols(f"D{i}")
    Dj = [sp.symbols(f"D{j}") for j in j_indices]
    
    #Funtion definition for Generating function for Known vector quantity Pi or Qi
    def generate_p_function(Vi, Vj, Di, Dj):
      summation_term = sum(Vj[k] * np.abs(Y[i-1, k]) * sp.cos(np.angle(Y[i-1, k]) + Dj[k] - Di) for k in range(len(Vj)))
      full = Vi * summation_term
      return full
    
    def generate_q_function(Vi, Vj, Di, Dj):
      summation_term = sum(Vj[k] * np.abs(Y[i-1, k]) * sp.sin(np.angle(Y[i-1, k]) + Dj[k] - Di) for k in range(len(Vj)))
      full = -Vi * summation_term
      return full
      
    #Generating the function of the Known Quantity
    if qty.type == 'P':
      func = generate_p_function(Vi, Vj, Di, Dj)
    elif qty.type == 'Q':
      func = generate_q_function(Vi, Vj, Di, Dj)
    
    
    #Differentiating the function with respect to 
    for qty in Uvector.data:       
      var = sp.symbols(f"{qty.type}{qty.bus}")  # determining the independent variables for differentiation for each iteration
      diff = sp.diff(func, var)
      
      # Generating values for substitution 
      subs = {}
      
      for item in range(len(V)):      #Values of vector V
        sub = {f'V{item + 1}': V[item]}
        subs.update(sub)
        
      for item in range(len(D)):      #Values of vector Delta
        sub = {f'D{item + 1}': D[item]}
        subs.update(sub)
        
      eval_val = float(diff.subs(subs))
      JacobV = np.append(JacobV, eval_val)
      n = len(V)
      
  JacobM = JacobV.reshape(n, n)
  return JacobM


#updates the Uvector, the Delta vector and the V vector
def update(init, D, V):
    for obj in init.data:
      if obj.type == 'D':
        D[obj.bus -1 ] = obj.value
      elif obj.type == 'V':
        V[obj.bus -1] = obj.value



#This function converts values from polar form to rectangular form for calculations
#It takes arguments of the magnitude and angle in degrees
def rect(r, theta):
  real = r*np.cos(np.radians(theta))
  img = r*np.sin(np.radians(theta))
  result = real + img*1j
  return result

    
# transforms the initial vector object 
# into an array of values for mathematical computation
def arrify(vector):
  array = []
  for obj in vector.data:
    array.append(obj.value)
  return np.array(array) 

# transforms the resulting array of the computation 
# into the Uvector object to be returned
def vectorfy(array, vector):
  new_Vector = Uvector()
  i = 0
  for obj in vector.data:
    obj.value = array[i]
    new_Vector.push(obj)
    i += 1
  return new_Vector