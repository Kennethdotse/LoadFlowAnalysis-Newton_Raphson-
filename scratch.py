import sympy as sp

x, y, z = sp.symbols('x y z', real = True)

f = 2*x**3

df_dx = sp.diff(f, x)

result = df_dx.subs({z: 3, y: 5, x:2})
print(result)

