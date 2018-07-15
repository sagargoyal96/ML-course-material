import sys
import os
import numpy as np
import csv
# import plotly.plotly as py
# import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



BATCH_SIZE=500
LEARN_RATE=0.009



def load_x_lr():
	global rownum
	file_x=open('linearX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	rownum_x=0
	for row in reader:
		# [1,row]
		X_dat.append([1,float(row[0])])
		rownum_x+=1

	file_x.close()

# Reading the Y file


def load_y_lr():
	file_y=open('linearY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0



	for row in reader:
		# [1,row]
		Y_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()

#  Oj =Oj+ (yi-hxi)/m * xj 
# end-start= batch size
def load_weigh_x():
	file_x=open('weightedX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	row_x=0
	for row in reader:
		# [1,row]
		weightX_dat.append([1,float(row[0])])
		row_x+=1

	file_x.close()

def load_weigh_y():
	file_y=open('weightedY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0



	for row in reader:
		weightY_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()	

def load_log_x():
	file_x=open('logisticX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	row_x=0
	for row in reader:
		# [1,row]
		logX_dat.append([1,float(row[0]),float(row[1])])
		row_x+=1

	file_x.close()

def load_log_y():
	file_y=open('logisticY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0



	for row in reader:
		logY_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()	
# performs normal gradient descent, by finding iteration loss 

def desc(X_,Y_):

	Ttrans_x=np.matmul(T,np.transpose(X_))
	matri=np.matmul((Y_-Ttrans_x),X_)
	temp=np.matmul(X_,np.transpose(T))-Y_
	loss=np.matmul(np.transpose(temp),temp)
	return matri[0],matri[1],loss


# not used yet

def desc_with_batch(start, end):
	i=start
	summ0=0
	summ1=0
	loss=0
	while(i<end):
		Ttrans_x=np.dot(T,X_data[i])
		diff_loss=Y_data[i]-Ttrans_x
		loss+=(diff_loss*diff_loss)/2
		loss0=diff_loss*X_data[i][0]
		loss1=diff_loss*X_data[i][1]

		summ0+=loss0
		summ1+=loss1
		i+=1

	loss=loss/(end-start)
	summ0=summ0/(end-start)
	summ1=summ1/(end-start)
	return summ0,summ1,loss

def plot_data_logistic(theta):
	tempx=[]
	tempy=[]
	for item in logX_data:
		tempx.append(item[1])
		tempy.append(item[2])
	i=0
	for item in tempx:
		pl.plot(tempx[i],tempy[i],'ro',c= ('C2' if logY_data[i] == 0 else 'C1'))
		i+=1
	
	x = np.linspace(0,20,1200)
	pl.plot(x,theta[0,1]*(-1)/theta[0,2]*x+theta[0,0]*(-1)/theta[0,2])
	pl.show()


def plot_data(theta):
	pl.plot(X_data,Y_data,'ro')
	x = np.linspace(0,20,1200)
	pl.plot(x,theta[1]*x+theta[0])
	pl.show()

def plot_weighted(theta):

	fitx=[]
	fity=[]
	for item in theta:
		fitx.append(item[0])
		fity.append(item[1])

	# print(theta)

	pl.plot(weightX_data[:,1],weightY_data,'ro')
	# x = np.linspace(0,20,1200)
	pl.plot(fitx,fity)
	pl.show()

# runs epochs till the loss obtained is very less , calls the method desc that computes the loss
def grad_desc(X_,Y_):
	while(1):
		i=0
		loss=0
		# while(i<rownum_x):
		# 	if i+BATCH_SIZE<rownum_x:
		# des0,des1,loss=desc(X_data,Y_data)

		des0,des1,loss=desc(X_,Y_)
				# i+=BATCH_SIZE
			# else:
				# des0,des1,loss=desc()
				# i=rownum_x
			# print(des0, des1)
		T[0]=T[0]+LEARN_RATE * des0
		T[1]=T[1]+LEARN_RATE * des1

		X3.append(T[0])
		Y3.append(T[1])
		Z3.append(loss)
		print("loss= ",loss)

		if loss<0.001:
			break
	# print("loss=",loss)
	return loss
		# print(T)

	# print("jjjjjj\n\n\n")

# finds the value of theta using the normal equation and uses it to plot the curve
def desc_normaleq(X_,Y_):
	XtX=np.matmul(np.transpose(X_),X_)
	print(XtX)
	ans=np.matmul(np.matmul(np.linalg.inv(XtX),np.transpose(X_)),Y_)
	return ans

# computes the hx for logistic regression (after applying the functionover the entire matrix and returns the value

def calc_log_hx(X_):
	ans1=np.matmul(log_T,np.transpose(X_))
	ans=np.apply_along_axis(my_func,0,ans1)
	return ans

# this is called by calc_log_hx to compute the value of hx
def my_func(a):
	return 1/(1+np.exp(-1*a))

# computes the Hessian and the Grad J(o) and uses it to calc the loss in logistic regression
def log_reg(X_, Y_):
	global log_T
	grad_J=np.matmul((np.matrix((Y_-calc_log_hx(X_)))),np.matrix(X_))
	
	# print(np.shape(X_))
	# print(np.shape(grad_J))
	i=0

	H=[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
	for item in X_:
		ith=np.matmul(np.transpose(np.matrix(X_[i])),np.matrix(X_[i]))

		hx=my_func(np.matmul(np.matrix(log_T),np.transpose(np.matrix((X_[i])))))
		# print("hx= ",hx)
		fin_ith=np.multiply((hx*(1-hx)),ith)
		H=np.add(H,fin_ith)
		i+=1

	Hinv=np.linalg.inv(H)
	change=np.matmul(Hinv,np.transpose(grad_J))
	# print(np.shape(change))
	change=np.multiply(LEARN_RATE,np.transpose(change))
	log_T=np.add(log_T,change)	

	return change


# runs epochs till the loss reduces beyond a certain value for logistic regression
def log_reg_loop(X_,Y_):
	count=0
	while(1):
		chng=log_reg(X_,Y_)
		# print(chng)
		count+=1
		if abs(chng[0,0])<0.00001 and abs(chng[0,0])<0.00001:
			break;

def create_w(x,X_):
	dev_arr=[]
	count=0
	for item in X_:
		count+=1
		temp=np.exp((-1)*np.power((x-item),2)/(2*0.8*0.8))
		dev_arr.append(temp)

	w=np.zeros((count,count), float)
	n_cou=0
	for item in dev_arr:
		w[n_cou][n_cou]=item
		n_cou+=1

	return w

def weighted_lr(X_, Y_):
	xs=[]
	for item in X_:
		xs.append(item[1])
	xs_=np.array(xs)

	# print(xs_)
	maxi=np.amax(xs_, axis=0)
	mini=np.amin(xs_, axis=0)

	# print("\n\n",maxi)

	points=np.linspace(mini,maxi,num=100)

	curve=[]

	for item in points:
		W=create_w(item,xs_)
		# print(W)
		# print(W)
		The=np.linalg.inv(np.matmul(np.matmul(np.transpose(X_),W),X_))
		Oth=np.matmul(np.matmul(np.transpose(X_),W),Y_)
		Theta=np.matmul(The,Oth)
		y=Theta[1]*item + Theta[0]
		curve.append([item,y])

		# print(Theta)

	# print(curve)
	return curve





T=np.array([0.0,0.0])
log_T=np.array([0.0,0.0,0.0])

X_dat=[]
rownum_x=0
Y_dat=[]
weightX_dat=[]
weightY_dat=[]
logX_dat=[]
logY_dat=[]


load_x_lr()
load_y_lr()
load_weigh_y()
load_weigh_x()
load_log_y()
load_log_x()

# pp=[[1,2],[3,4],[1,5]]
# print(calc_log_hx(pp))

# convert to numpy arrays

X_data=np.array(X_dat)
Y_data=np.array(Y_dat)
weightX_data=np.array(weightX_dat)
weightY_data=np.array(weightY_dat)
logY_data=np.array(logY_dat)
logX_data=np.array(logX_dat)

X3=[]
Y3=[]
Z3=[]

# curve=weighted_lr(weightX_data,weightY_data)
# plot_weighted(curve)



log_reg_loop(logX_data,logY_data)

plot_data_logistic(log_T)


# lss=grad_desc(X_data,Y_data)

# tt=desc_normaleq(weightX_data,weightY_data)


# # print(tt)
# # plot_data(tt)


		






































