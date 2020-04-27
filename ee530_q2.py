import sympy
from scipy import optimize
from sympy import *
import numpy as np
import types
import matplotlib.pyplot as plt
import time

def connect_points(x1,x2,y1,y2):
    plt.plot([x1,x2],[y1,y2],'k-')
    
def phi(x0,y0,a,d_vec):
    x = x0+(a*d_vec[0]) 
    y = y0+(a*d_vec[1])
    return (1-x)**2 + (y-(x**2))**2
    #return f.subs([(x,x0+a*d_vec[0]),(y,y0+a*d_vec[1])])
def diff(x0,y0,alpha,d_vec):
    x = x0+(alpha*d_vec[0])
    print(d_vec)
    y = y0+(alpha*d_vec[1])
    return (2*(x-1) - (4*(y-(x**2))*x)) , 2*(y-(x**2))

def cal_alpha(x0,y0,d_vec,alpha):

    grad_x,grad_y = diff(x0,y0,alpha,d_vec) #partial derivative at x0+alpha*d,y0+alpha*d
    grad = [grad_x,grad_y]
    grad = np.array(grad) #change to numpy array
    sol = solve(np.dot(grad,d_vec),alpha) #returns array of all possible solutions
    sol = sympy.sympify(sol)
    sol1=[]
    
    for itr in range(len(sol)):
        real = N(sol[itr],chop=True)
        sol1.append(real)
    sol1 = np.array(sol1)
    print(sol1)
    sol2 = []
    mini =  []
    for itr in range(len(sol1)):
        if sol1[itr].is_real:
            sol2.append(sol1[itr])
            mini.append(phi(x0,y0,alpha,d_vec).subs(alpha,sol1[itr]))
    print(mini)
    mini = np.array(mini)
    min_index = np.argmin(mini)
    return sol2[min_index]
     

def calc_grad(x0,y0):
    grad = [2*(x0-1)-(4*(y0-(x0**2))*x0),2*(y0-(x0**2))]
    grad = np.array(grad, dtype=np.float64) #convert to numpy array
    mag = np.linalg.norm(grad) #calculate norm
    return grad,mag

start_time = time.time()
alpha = sympy.symbols('alpha') #create symbol alpha
x_old,y_old = 1,2   
itr = 0
mag = 10

fig=plt.figure(figsize=(6,5))
l,b,w,h = 0.1,0.1,0.8,0.8
ax = fig.add_axes([l,b,w,h])
x = np.linspace(-1,2,num=50)
y = np.linspace(-1,2,num=50)
x_points,y_points = np.meshgrid(x,y)  
#f1 = lambdify([x,y],f,'numpy')
levels = (1-x_points)**2+(y_points-(x_points**2))**2
levels = np.array(levels)
c = ax.contour(x_points,y_points,levels,50)
ax.scatter(x_old,y_old,c='r',marker="*")
ax.annotate('(x0,y0)',xy=(x_old,y_old),textcoords='offset points')
ax.set_title('Rosenbrock function using optimal gradient descent')
while mag > 1e-3:
    grad,mag = calc_grad(x_old,y_old)
    if(mag > 1e-3):
        dir_vec = (-grad/mag).T
    else:
        break 
    alpha1 = cal_alpha(x_old, y_old,dir_vec,alpha)
    x_new = x_old + (alpha1*dir_vec[0])
    y_new = y_old + (alpha1*dir_vec[1])
    connect_points(x_new,x_old,y_new,y_old)
    x_old = x_new
    y_old = y_new
    ax.scatter(x_old,y_old,c='r',marker="*")
    itr+=1
print("--- %s seconds ---" % (time.time() - start_time),'no of steps=' ,itr)    
#print(mag,' ',itr)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
