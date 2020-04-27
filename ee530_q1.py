import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()
def connect_points(x1,x2,y1,y2):
    plt.plot([x1,x2],[y1,y2],'k-')
alpha = 0.003
x_k , y_k = 1, 2
itr = 0
mag = 10
fig=plt.figure(figsize=(6,5))
l,b,w,h = 0.1,0.1,0.8,0.8
ax = fig.add_axes([l,b,w,h])
x_ = np.linspace(-1,2,num=50)
y_ = np.linspace(-1,2,num=50)
x,y = np.meshgrid(x_,y_)
levels = (x-1)**2 + (y-(x**2))**2
c = ax.contour(x,y,levels,50)
ax.scatter(x_k,y_k,c='r',marker="*")
ax.annotate('(x0,y0)',xy=(x_k,y_k),textcoords='offset points')
ax.set_title('Rosenbrock function using fixed step size = 0.003 and (x0,y0) = (2,1)')
while mag > 1e-3:
 	grad_x = 2*(x_k-1)+(4*(x_k**2-y_k)*x_k)
 	grad_y = 2*(y_k-(x_k**2))
 	grad = [grad_x,grad_y]
 	mag= np.linalg.norm(grad) 
 	x = x_k - (alpha*grad_x/mag)
 	y = y_k - (alpha*grad_y/mag)
 	connect_points(x,x_k,y,y_k)
 	x_k = x
 	y_k = y
 	if(itr%5==0):
 	#	print(x,' ',y)
 		ax.scatter(x,y,c='r',marker="*")
 	itr+=1
print(x,' ',y,' ',mag, ' ',itr)	 

print("--- %s seconds ---" % (time.time() - start_time))
plt.xlabel('x')
plt.ylabel('y')	
plt.show()
