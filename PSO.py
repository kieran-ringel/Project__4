import numpy as np
import random as rand
import math
from Particle import Particle as part
from Feedforward import Feedforward as ff

class PSO:
    def __init__(self, initialNN, population, epochs, train):
        self.initialNN = initialNN
        return (PSO.pso(self, self.initialNN, population, epochs, train))
    def fitness(self, value, expected): #This function calculates the fitness of a particle (NN)
        error = 0
        if self.classification == 'classification':
            for node in range(len(value)):
                outputindex = value.index(value[node])
                inputindex = self.classes.index(expected)
                if outputindex == inputindex:
                    rt = 1
                if outputindex != inputindex:
                    rt = 0
                if value[node] == 0:
                    value[node] = 0.0001
                
                error -= (rt * math.log10(abs(value[node])))
                
        if self.classification == 'regression':
            error = ((float(value[0]) - float(expected)) ** 2) /2
        return (error)
        
    def pso(self, initialNN, population,epochs,train):
        
        
        inertia = 15 ##inertia to be tuned
        cog = 3.6 ##cognitive componet to be tuned
        soc = 7.4##social componenet to be tuned
        
        constrict = 3.8 #constriction coefficient to be tuned
        exiterror = 54.2 #so we are not stuck in loop forever.Controls while loop
        globalBestP = self.initialNN[0] #initial the best positon to a random NN
        globalBestE = 10000000 #the best error
        swarm = [0] * self.population #a population of NN
        
        for particle in range(len(swarm)):#same as population of NN, a population of NN is a swarm
            self.NN = self.initialNN[particle] #a particle
            
            position = [] #holds the position of the particle
            numw = 0
            for layer in range(len(self.NN)):
                for node in range(len(self.NN[layer])):
                    for weight in range(len(self.NN[layer][node])):
                        numw+=1
                        position.append(self.NN[layer][node][weight])
            
            fit = 0 #calculates fitness for every NN
            for i in range(epochs):
                train = train.sample(frac=1).reset_index(drop=True)
                
                for row, trainpoints in train.iterrows():    
                   
                   node_values = ff.feedforward(self, trainpoints)
                   fit += PSO.fitness(self, node_values[-1], trainpoints['class'])
            fit /=epochs * len(train)
            velocity = [0] * numw
            for w in range(numw):#computes velocity for the position of the particle
                velocity[w] = rand.uniform(0,0.1)
            swarm[particle] = part(position, fit, velocity, position, fit) #creates a particle object, which will hold info associated with particle
            if swarm[particle].fitness<globalBestE: #if fitness is better than global fitness, update fitness to new fitness and new best position is the associated position
                globalBestE = swarm[particle].fitness 
                globalBestP = swarm[particle].position
        
      
        while(globalBestE>exiterror):
            
            
            for particle in range(len(swarm)):
                fit = 0
                for i in range(epochs):#calculates fitness of position
                    train = train.sample(frac=1).reset_index(drop=True)
                    for row, trainpoints in train.iterrows():    
                        node_values = ff.feedforward(self, trainpoints)
                        fit += PSO.fitness(self,node_values[-1], trainpoints['class'])
                fit /= epochs * len(train)
                swarm[particle].fitness = fit
                
                if swarm[particle].fitness<swarm[particle].bestfit: #if fitness is better than previous personal fitness, update to new fitness and new position
                    swarm[particle].bestfit = swarm[particle].fitness
                    swarm[particle].bestposition = swarm[particle].position
                if swarm[particle].fitness<globalBestE:#if fitness is better than global fitness, update fitness to new fitness and new best position is the associated position
                    globalBestE= swarm[particle].fitness
                    globalBestP = swarm[particle].position
                    
                newV = [0] * numw
                newP = [0] * numw
                for w in range(numw): #calculating new velocity and position
                    beta1 = cog * rand.uniform(0, 1)
                    beta2 = soc * rand.uniform(0,1)
                    betatot = beta1 + beta2
                    aval = abs(betatot * (betatot -4))
                    sqr = math.sqrt(aval)
                    X = (2 * constrict) / aval #this is using the fancy constriction coefficent to help PSO converge :)
                    
                    newV[w] = X *((inertia * swarm[particle].velocity[w]) + (beta1 * (swarm[particle].bestposition[w] - swarm[particle].position[w])) + (beta2 * (globalBestP[w] - swarm[particle].position[w])))#CALCULATE VELOCITY
                    newP[w] = swarm[particle].position[w] + newV[w]


                swarm[particle].velocity = newV #updating new velocity
                swarm[particle].position = newP #updating new position
        
        return(globalBestP) 
                   
            
            

                
            

       

        
       
            
          
            
                
                
            
        
       
                
      
      
        
        

      
      
        

        
        
