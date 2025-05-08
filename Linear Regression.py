import numpy as np
import matplotlib.pyplot as plt

#Loading The Data
x_train = np.array([1.0,2.0]) #Features
y_train = np.array([300.0,500.0])  #Target Value

#Function To Calc Cost
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = cost + (f_wb - y[i])**2
        Total_cost = 1/(2*m) * cost

    return Total_cost
    
#Calc Derivative 
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i]+b

        #Calc Partial Derivative
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i

        dj_dw = dj_dw / m
        dj_db = dj_db / m

    return dj_dw,dj_db
    

#Calc Gradient Descent

def compute_gradient_descent(x, y,w,b, alpha, num_iters, gradient_function):
    for i in range(num_iters):
        # Compute gradients
        dj_dw,dj_db = gradient_function(x,y,w,b)

        #Update parameters
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        return w,b
    

#Inital Parameters
w_init = 0
b_init = 0
iterations = 10000
alpha = 1.0e-2
w_final, b_final = compute_gradient_descent(x_train, y_train, w_init, b_init,alpha,iterations, compute_gradient)
print(f"(w,b) found by gradient descent:{w_final:8.4f}, {b_final:8.4f} ")

#Predictions

print(f"1000 sqft house prediction {w_final*1.5 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2500 sqft house prediction {w_final*2.5 + b_final:0.1f} Thousand dollars")