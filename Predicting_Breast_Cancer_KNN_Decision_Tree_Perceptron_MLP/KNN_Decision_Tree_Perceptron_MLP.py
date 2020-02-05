import math
import numpy as np

class KNN:
	def __init__(self, k):
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.y = y
		self.X = X
		
	def predict(self, X):
		#Defining an array of predictions where there is one prediction for each set of features
		predictArr = []
		# Running for loop for the testing dataset
		# In the tarining dataset, calculating the distance b/w all the examples & the testing example
		for value in X:
			#Defining the storeTmporary array storing predictions
			predictstoreTmp = []
			#Running for loop for each X and y value
			for i,j in zip(self.X,self.y):
				#appending the value in the predictstoreTmp array
				predictstoreTmp.append([j,self.distance(value,i)])
				# distance categorization
			predictstoreTmp.sort(key = lambda value : value[1])
			# initializing a variable 'p' for while loop
			p = 0
			#Defining and initializing variable to count the number of 1 labels
			cntOne = 0
			#Defining and initializing variable to count the number of 0 labels
			cntZero = 0
			#Running while loop for the range of 'k'
			while p in range(self.k):
				#checking condtion, if the first column element is zero or not
				if predictstoreTmp[p][0] != 0:
					cntOne = cntOne + 1
				else:
					cntZero = cntZero + 1
				#incrementing 'p' by one
				p = p+1
			# if more examples with label '1', predict '1' otherwise predict '0'
			if cntOne < cntZero:
				# self.predictArr = [*self.predictArr, 0]  
				predictArr.append(0)	
			else:
				predictArr.append(1)
			#returning the predictArr			
		return np.array(predictArr)

class ID3:
	def __init__(self, nbins, data_range):
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		valCategorized = np.floor(self.bin_size*norm_data).astype(int)
		return valCategorized

	def decisionTree(self,X,y,features,revert):
		# Calculating the distinctive counts
		distinctive, cnts = np.unique(y, return_counts=True)
		md = np.argmax(cnts)
		# if there is not any example present then it returrn the parent's mode
		if y.shape[0] == 0:
			return revert
		# else if every example is in one same class, then return the labal of the class
		elif np.unique(y).size == 1:
			return y[0]
		#else if returnn the class's mode in case, there are no features
		elif len(features) == 0:
			return md
		else:
			#array to save every feature's info gain that is i/p to the func 
			infoGain = {}
			# using for loop above features
			for f in features:
				# 'total' - a varible to save the total addition of the weighed calcEntropy of categeries of an attrib
				total = 0
				#calculating the no. of ex relating to eaach classs
				cal_ex = [0,0]
				#array to save the no. of exm relating to every class for attributs' every categry 
				storeTmp = {}
				# using for loop for every example
				for iteretor,iterery in zip(X[:,f],y):
					#using if condition on whether iteretor is in temporary stoarge or not
					if iteretor not in storeTmp:
						storeTmp[iteretor] = [0,0]
					# increase the count of one by one, in case, the ex's labal is one
					if iterery == 1:
						storeTmp[iteretor][1] = 1 + storeTmp[iteretor][1]
						cal_ex[1] = 1 + cal_ex[1]	
					# increase the count of zero by one, in case, the ex's labal is zero
					elif iterery == 0:
						storeTmp[iteretor][0] = 1 + storeTmp[iteretor][0]
						cal_ex[0] = 1 + cal_ex[0]
				# calculating the entry of every attrib
				entry_attrib = self.calcEntropy(cal_ex)
				# using for loop to to compute the entrpy of every categry of an attrib
				for q in storeTmp:
					calcEntropy_val = self.calcEntropy(storeTmp[q])
					#Updating the total
					total = calcEntropy_val*sum(storeTmp[q]) + total
				# getting the information gain when the node is split on the attribute being considered and storing it in the dictionary ents
				totBySum = total/sum(cal_ex)
				infoGain[f] = entry_attrib - totBySum
			#storing the attrib that provides the best info. gain
			bestInfoGain = sorted(infoGain.items(), key=lambda kv: kv[1], reverse = True)
			#storing the 1st attribute of bestInfoGain in 'attribBest'
			attribBest = bestInfoGain[0][0]
			# Making a node for attribBest
			baseNode = Node(attribBest)
			# discarding the attrib that was used to partition the list of present attribs
			features.remove(attribBest)
			# using for loop to recursivly compute the attribs to partition on every categary's node and making the forming subtreee after that
			for d in np.unique(X[:,attribBest]):
				# saving the ex that are related to the categary of the chosen attrib
				# Making array 'y_d' 
				y_d = [];
				# Making array 'X_d'
				X_d = [];
				#for loop on iterater, iterery in X,y
				for iteretor,iterery in zip(X,y):
					#if condtion to check whether iteretor[attribBest] is equal to 'd' or not 
					if iteretor[attribBest] == d:
						#Appending the iterery in y_d
						y_d.append(iterery)
						#Appending the iteretor in X_d
						X_d.append(iteretor)			
				# Making a subtreee for a class node of an attrib
				attribSubtree = self.decisionTree(np.array(X_d),np.array(y_d),features.copy() ,md)
				# appending the attrib to baseNode
				baseNode.childrn[d] = attribSubtree
				# storing the mode of the examples in the node
				baseNode.mode = md
			#Returning the baseNode
			return baseNode

	# the calcEntropy method gives the list every class's count, basically 1 and 0 countt
	def calcEntropy(self,cnt):
		# Assigning the length of count to 'length_cnt' variable
		length_cnt = len(cnt)
		#initializing and defining the variable 'total'
		total = 0
		#Running for loop for a in the range 0 to the length of count
		for i in range(0, length_cnt):
			#taking the sum of counts
			sum_cnt= sum(cnt)
			cnt_by_sum= cnt[i]/sum_cnt
			#if count of ex. in relation to the class is equal to 0, then execute the break and come out of the loop
			if cnt_by_sum != 0:
				break
			else:
				continue
			# Calculating the log of cnt_by_sum to the base 2
			logc = math.log(cnt_by_sum,2)
			# Updating the total
			total = total - (logc*cnt_by_sum)
		#Returning the total calculated
		return total

	def train(self, X, y):
		# Calculating the distinctive counts
		distinctive, cnts = np.unique(y, return_counts=True)
		#assigning value to the mode variable
		md = np.argmax(cnts)
		#Getting the categorized data from the preprocess method
		valCategorized = self.preprocess(X)
		#Getting the attributes
		features = [i for i in range(0, (X.shape[1]))]
		#formation of the descision Tree
		self.dsnTree = self.decisionTree(valCategorized,y,features,md)

	def predict(self, X):
		# array to save the predictons
		storePredicts = []
		#Return array of predictions where there is one prediction for each set of features
		valCategorized = self.preprocess(X)
		# using for loop to go over the ex
		for i in valCategorized:
			# base node
			storeTmp = self.dsnTree
			#retrieving the catagory of the attrib
			presentVal = i[storeTmp.pointer]
			#Running while
			while 1:
				#if condition to check whether presentVal in inside the childrn of node
				if presentVal in storeTmp.childrn:
					#check whether the childrn correlated to the category is labal and actually not the nod, in that case, it anticipate the classs
					if storeTmp.childrn[presentVal] == 1:
						storePredicts.append(storeTmp.childrn[presentVal])
						break
					elif storeTmp.childrn[presentVal] == 0:
						storePredicts.append(storeTmp.childrn[presentVal])
						break
					#updating the presentVal, storeTmp, and loop, in case, the childrn related to the categary is a subtreee
					storeTmp =  storeTmp.childrn[presentVal]
					#Assigning value to the presentVal variable
					presentVal = i[storeTmp.pointer]
				# in case, the presentVal is not inside the childrn then the anticipated labal is basically the mod of the ex in nod
				elif presentVal not in storeTmp.childrn:
					#Appending the value
					storePredicts.append(storeTmp.mode)
					break
		# Returning the storePredicts array
		return np.array(storePredicts)

class Node:
	def __init__(self, pointer):
		self.pointer = pointer
		# node's children
		self.childrn = {}

class Perceptron:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#Defining and initializing a count variable 
		cnt = 0
		#Calculating the length variable value
		length = int(steps/y.size)
		#Running while loop on cnt variable
		while cnt in range(0, length):
			#Running for loop for every example of tarining
			for p,q in zip(X,y):
				# computing the output of the perceptron
				solution = self.b + np.matmul(p,self.w)
				#if-else condition on solution
				if solution <= 0:
					solution = 0
				elif solution > 0:
					solution = 1
				# if the predicted output does not match the actual output, we update the weights
				if q!=solution:
					self.w = self.w + self.lr*(p*q)
			#incrementing counter of while loop
			cnt = cnt + 1

	def predict(self, X):
		# Computing the prediction array where for every set of features, there is a single prediction
		solution = np.matmul(X,self.w) + self.b
		solution[solution <= 0] = 0
		solution[solution>0] = 1
		#Returning the solution which is the prediction array	
		return solution

# defining the settngs to ignore the warnings, in case, there is an underflo or overflo
reinstateSetings = np.geterr()
errorSetings = reinstateSetings.copy()
errorSetings["under"] = "ignore"
errorSetings["over"] = "ignore"

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def forward(self, input):	
		z = np.matmul(input,self.w) + self.b
		# we are saving 'x' which is the i/p for the backward pass for this layer
		self.value = input
		# Returning the value of 'z'
		return z

	def backward(self, gradients):
		#calculating the transpose of weight matrix
		nptrnpsw = np.transpose(self.w)
		value2 = np.matmul(gradients, nptrnpsw)
		#calculating the transpose of value
		nptrnpsval = np.transpose(self.value)
		value1 = np.matmul(nptrnpsval,gradients)
		#utilizing the learnig rate 'lr', calculating the new uptodate of weight matrix
		w = self.w - value1*self.lr
		#utilizing the learnig rate 'lr', calculating the new uptodate of bias matrix 
		b = self.b - gradients*self.lr
		# Returning value2
		return value2

class Sigmoid:
	def __init__(self):
		None

	def forward(self, input):
		#Ignore warnigs of overflo and underflo
		np.seterr(**errorSetings)
		# we are saving the i/p for the bacward pass
		self.save = input
		h= np.exp(-input) + 1
		#Defining the Sigmoid function
		sigmd = 1/h
		# Reinstating the settings
		np.seterr(**reinstateSetings)
		#Returning the sigmoid function
		return sigmd

	def backward(self, gradients):
		d = self.forward(self.save)
		e = 1-d;
		return gradients*d*e