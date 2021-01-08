import numpy as np
import matplotlib.pyplot as plt
##################################################################
######################Problem1####################################
##################################################################
"""
    Define function
"""
def gaussian_data_generator(mean,var):
    '''
        input mean and var
        Z~gaussian(0,1)-> Z=X-mean / std    
        X~gaussian(mean,var) -> X=Z*std+mean
    '''
    std=np.sqrt(var)
    random_data=np.sum(np.random.uniform(size = 12))-6
    #------------------------------------
    #------------calculate random data---
    #------------------------------------
    print('\n')
    random_data=random_data*std+mean
    return(random_data)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def computeDotProduct(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    dotProduct = 0
    for i in range(n):
        dotProduct += a[i] * b[i]
    return dotProduct

def computeHessianMatrix(data, hypothesis):
    hessianMatrix = []
    n = len(data)
 
    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-hypothesis)*hypothesis)
        hessianMatrix.append(row)
    return hessianMatrix

def computeVectPlus(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum

def computeTimesVect(vect, n):
    nTimesVect = []
    for i in range(len(vect)):
        nTimesVect.append(n * vect[i])
    return nTimesVect

def Plot_Confusion_table(TP,FN,FP,TN):
    print('      Confusion table')
    print('                      actual label 0  actual label 1')
    print('        pred_label 0        %d(TP)          %d(FP)                red point: Label0' %(TP,FP))
    print('        pred_label 1        %d(FN)          %d(TN)                blue point: Label1'%(FN,TN))
    sensitivity=TP/(TP+FN)
    Specificity=TN/(FP+TN)
    print('\n')
    print('sensitivity (Successfully predict Label0): %f'%(sensitivity))
    print('Specificity (Successfully predict Label0): %f'%(Specificity))
    
    
    
#########################################################
#####################Input and Setting###################
#########################################################
"""
    Input Setting:
        D1={(x1,y1),.....,(xn,yn)}              D2={(x1,y1),.....,(xn,yn)}
        x is from N(mx1,vx1)                    x is from N(mx2,vx2)
        y is from N(my1,vy1)                    y is from N(my2,vy2)
"""
print("Enter nums of data points:")  
nums=int(input())                #nums of data points

####D1####
print("Enter mx1:")
mx1=int(input())                #mx1
print("Enter vx1:")
vx1=int(input())                #vx1
print("Enter my1:")
my1=int(input())                #my1
print("Enter vy1:")
vy1=int(input())                #vy1


####D2####
print("Enter mx2:")
mx2=int(input())                #mx2
print("Enter vx2:")
vx2=int(input())                #vx2
print("Enter my1:")
my2=int(input())                #my2
print("Enter vy1:")
vy2=int(input())                #vy2

"""
    Producing D1 & D2
"""
D1,D2=[],[]
for i in range(nums):
    D1_data_point=[]
    D2_data_point=[]
    
    D1_x=gaussian_data_generator(mx1,vx1)
    D1_y=gaussian_data_generator(my1,vy1)
    D2_x=gaussian_data_generator(mx2,vx2)
    D2_y=gaussian_data_generator(my2,vy2)
    
    D1_data_point.append(D1_x)
    D1_data_point.append(D1_y)
    D2_data_point.append(D2_x)
    D2_data_point.append(D2_y)
    D1.append(D1_data_point)
    D2.append(D2_data_point)
D1,D2=np.array(D1),np.array(D2)

"""
    Plotting
    Data_X: store D1+D2 data point
"""
D1_x,D1_y,D2_x,D2_y=[],[],[],[]
for i in range(nums):
    D1_x.append(D1[i][0])
    D1_y.append(D1[i][1])
    D2_x.append(D2[i][0])
    D2_y.append(D2[i][1])
plt.figure() #同時顯示多張圖
plt.plot(D1_x, D1_y, '+', label='D1', color='red')
plt.plot(D2_x,D2_y,'o',label='D2',color='blue')
plt.title('GroundTruth')


"""
    -> Producing X(nums*d+1): (50+50)*(2+1)  #2->feature 數量
        theta:[w0 w1 w2].transpose
        [[1,x1,y1]
        [1,x2,y2]
        [1,x3,y3]  = X 
        .......
        [1,xn,yn]]
        
    -> Producing Label : Y 1*100
        red-> 0 , blue-> 1
        [0,.....,0,1,1,.....,1] = Label  
"""
###X###
Data_X=np.vstack((D1,D2))  # Type -> np.array
add_one=np.array([[1 for i in range(nums*2)]])
Data_X=np.column_stack((add_one.T,Data_X))
#print(Data_X)
###Label###
Label=[]               #Type -> List
for i in range(nums):
    Label.append(0)
for i in range(nums):
    Label.append(1)
#print(Label)
    
#########################################################
#####################Newton's Method#####################
#########################################################
m,n=np.shape(Data_X)  # m: data nums(100), n: feature nums+1 (2+1)
theta=[0]*n
iternum=30
while(iternum):
    gradientSum=[0]*n
    hessianMatSum=[[0]*n]*n
    for i in range(m):
        try:
            hypothesis=sigmoid(computeDotProduct(Data_X[i], theta))
        except:
            continue
        error=Label[i]-hypothesis
        gradient=computeTimesVect(Data_X[i], error/m)
        gradientSum = computeVectPlus(gradientSum, gradient)
        hessian = computeHessianMatrix(Data_X[i], hypothesis/m)
        for j in range(n):
                hessianMatSum[j] = computeVectPlus(hessianMatSum[j], hessian[j])
    try:
        hessianMatInv = np.mat(hessianMatSum).I.tolist()
    except:
        continue
    for k in range(n):
        theta[k] -= computeDotProduct(hessianMatInv[k], gradientSum)
    iternum-=1
print('w=',theta)
print('\n')

"""
    Calculate Confusion matrix:
                    GT_class0  GT_class1
        pre_class0     TP         FP
        pre_class1     FN         TN
        
    y=w0+w1x+w2y , x1->feature1 x2->feature2
    
    pred_D1 []
    pred_D2 []
"""
pred_D1x,pred_D1y,pred_D2x,pred_D2y=[],[],[],[]
TP,FN,FP,TN=0,0,0,0
for i in range(nums*2):
    pred_y=theta[0]+theta[1]*Data_X[i][1]+theta[2]*Data_X[i][2]
    if pred_y<0:
        pred_label=0
        pred_D1x.append(Data_X[i][1])
        pred_D1y.append(Data_X[i][2])
    else:
        pred_label=1
        pred_D2x.append(Data_X[i][1])
        pred_D2y.append(Data_X[i][2])
    actual_label=Label[i]
    #判斷屬於TP,FN,FP,TN    
    if actual_label==0 and pred_label==0:
        TP+=1
    elif actual_label==0 and pred_label==1:
        FN+=1
    elif actual_label==1 and pred_label==0:
        FP+=1
    elif actual_label==1 and pred_label==1:
        TN+=1
#Draw Confusion_table
print('Newton Confusion table:')
Plot_Confusion_table(TP,FN,FP,TN)
        
"""
    Plotting
"""
plt.figure() #同時顯示多張圖
plt.plot(pred_D1x, pred_D1y, '+', label='D1', color='red')
plt.plot(pred_D2x,pred_D2y,'o',label='D2',color='blue')
plt.title('Newton')

#########################################################
#####################End of Newton ######################
#########################################################


#########################################################
#####################Gradient Dscent#####################
#########################################################
"""
    Setting
    Data_matrix -> Data_X
    Labelmat -> Label
"""

Data_matrix=Data_X
Labelmat=Label
_theta=[0]*n
_m,_n=np.shape(Data_matrix)
_iternum=500
alpha=0.01 #learning rate 

while(_iternum):
    _gradientsum=[0]*n
    for i in range(_m):
        _hypothesis=sigmoid(computeDotProduct(Data_matrix[i],_theta))
        error=_hypothesis-Label[i]
        _gradient=computeTimesVect(Data_matrix[i],error)
        _gradientsum=computeVectPlus(_gradientsum,_gradient)
    _gradientsum=computeTimesVect(_gradientsum,alpha)
    _gradientsum=computeTimesVect(_gradientsum,-1)  #轉成負的  下一行丟進去vector + 才會有的效果
    _theta=computeVectPlus(_theta,_gradientsum)
    _iternum-=1
print("\n")
print('-------------------------------------------') 
print('\n')
print('w=',_theta)

"""
    Calculate Confusion matrix:
                    GT_class0  GT_class1
        pre_class0     TP         FP
        pre_class1     FN         TN
        
    y=w0+w1x+w2y , x1->feature1 x2->feature2
    
    pred_D1 []
    pred_D2 []
"""  
_pred_D1x,_pred_D1y,_pred_D2x,_pred_D2y=[],[],[],[]
TP,FN,FP,TN=0,0,0,0
for i in range(nums*2):
    _pred_y=theta[0]+theta[1]*Data_matrix[i][1]+theta[2]*Data_matrix[i][2]
    if _pred_y<0:
        pred_label=0
        _pred_D1x.append(Data_X[i][1])
        _pred_D1y.append(Data_X[i][2])
    else:
        pred_label=1
        _pred_D2x.append(Data_X[i][1])
        _pred_D2y.append(Data_X[i][2])
    actual_label=Label[i]
    #判斷屬於TP,FN,FP,TN    
    if actual_label==0 and pred_label==0:
        TP+=1
    elif actual_label==0 and pred_label==1:
        FN+=1
    elif actual_label==1 and pred_label==0:
        FP+=1
    elif actual_label==1 and pred_label==1:
        TN+=1
#Draw Confusion_table
print('Gradient Descent Confusion table:')
Plot_Confusion_table(TP,FN,FP,TN)
        
"""
    Plotting
"""
plt.figure() #同時顯示多張圖
plt.plot(_pred_D1x,_pred_D1y, '+', label='D1', color='red')
plt.plot(_pred_D2x,_pred_D2y,'o',label='D2',color='blue')
plt.title('Gradient Descent')
