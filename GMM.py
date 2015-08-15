#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

__author__="Qinxue"

import numpy as np
import Matrix as ma
import Statistics as stat
import os

class GMM():

	def __init__(self,listx,znum,iteration,error):
		self.listx=listx
		self.iteration=iteration
		self.znum=znum
		self.dimension=len(self.listx[0])
		self.insnum=len(self.listx)
		self.initmodelparameters()
		self.error=error

	def __init__(self,filepath,decodex,znum,dimension,iteration,error):
		self.listx=None
		self.filepath=filepath
		self.decodex=decodex
		self.znum=znum
		self.dimension=dimension
		self.iteration=iteration
		self.insnum=self.countinsnum()
		self.error=error
		self.initmodelparameters()

	def getnextdoc(self,filepath):
		for f in os.listdir(filepath):
			filename=filepath+'/'+f
			if os.path.isfile(filename):
				yield filename
			elif os.path.isdir(filename):
				self.getnextdoc(filename)

	def getnextx(self):
		if self.listx ==None:
			for doc in self.getnextdoc(self.filepath):
				with open(doc) as fp:
					lines=fp.readlines()
					for line in lines:
						if self.decodex==None:
							yield self.defaultdecodex(line)
						else:
							yield self.decodex(line)
		else:
			for x in self.listx:
				yield x

	def getnextminibatchx(self):
		pass

	def defaultdecodex(self,line):
		retlist=[]
		for l in line.split():
			l=l.strip()
			retlist.append(float(l))
		return np.array(retlist)

	def initmodelparameters(self):
		self.listw=np.random.rand(self.insnum,self.znum)
		self.normalizelistw()
		self.theta=np.zeros(self.znum)
		self.miu=np.random.rand(self.znum,self.dimension)
		self.sigma=np.random.rand(self.znum,self.dimension,self.dimension)

	def countinsnum(self):
		num=0
		for i in self.getnextx():
			num+=1
		return num

	def train(self):
		for i in range(self.iteration):
			theta=self.theta.copy()
			self.updateM()
			self.updateE()
			if i%10==0 and self.residual(theta)<self.error:
				break

	def predict(self,x):
		prew=np.zeros(self.znum)
		for i in range(self.znum):
			prew[i]=stat.multidimensionalgaussian\
				(x,self.miu[i],self.sigma[i],self.dimension)*self.theta[i]
		return prew.argmax()

	def updateM(self):
		# update theta
		sumw=np.zeros(self.znum)
		for w in self.listw:
			sumw+=w
		self.theta=1.0/self.insnum*sumw
		# update miu
		for i in range(self.znum):
			sumwx=np.zeros(self.dimension)
			j=0
			for x in self.getnextx():
				sumwx+=self.listw[j][i]*x
				j+=1
			self.miu[i]=sumwx/sumw[i]
		# update sigma
		for i in range(self.znum):
			sumwxu=np.zeros(self.dimension*self.dimension).\
			reshape(self.dimension,self.dimension)
			j=0
			for x in self.getnextx():
				sumwxu+=self.listw[j][i]*ma.vTvmultipy(x-self.miu[i],x-self.miu[i])
				j+=1
			self.sigma[i]=sumwxu/sumw[i]

	def updateE(self):
		j=0
		for x in self.getnextx():
			sumpxw=0.0
			for i in range(self.znum):
				self.listw[j][i]=stat.multidimensionalgaussian\
				(x,self.miu[i],self.sigma[i],self.dimension)*self.theta[i]
				sumpxw+=self.listw[j][i]
			for i in range(self.znum):
				self.listw[j][i]/=sumpxw
			j+=1

	def normalizelistw(self):
		for j in range(self.insnum):
			self.listw[j]/=np.sqrt(np.dot(self.listw[j],self.listw[j]))

	def residual(self,theta):
		return stat.norm(theta-self.theta)
