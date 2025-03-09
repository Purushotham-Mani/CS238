import sys
import numpy as np


class QLearning:
    def __init__(self, file):
        if file==1:     
            self.gamma = 0.95
            self.Q = np.zeros((100,4))
            self.visit = np.zeros((100))
        elif file==2:  
            self.gamma = 1
            self.Q = np.zeros((50000,7))
            for i in range(self.Q.shape[0]):
                for j in range(self.Q.shape[1]):
                    p=i%500
                    v=i//500
                    if p<=465 or p>=471:
                        self.Q[i][j] = 25*np.sign((v-49)*(j-3))
            self.visit = np.zeros((50000))
        else: 
            self.gamma = 0.95
            self.Q = np.zeros((302020,9))
            self.visit = np.zeros((302020))

    def estimateQstar(self,file):
        with open(file,'r') as f:
            f.readline()
            i=0
            for line in f:
                s, a, r, sp = [int(x) for x in line.split(',')]
                self.visit[s-1]=1
                alpha = (1/(1+i)**0.75)
                self.Q[s-1][a-1] = self.Q[s-1][a-1] + alpha*(r + self.gamma*(np.max(self.Q[sp-1]))-self.Q[s-1][a-1])
                i=i+1        
        f.close()
        


def writePolicy(Q,outputfilename):
    with open(outputfilename, 'w') as f:
        for i in range(Q.shape[0]):
            f.write(f"{np.argmax(Q[i])+1}\n")
 

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.policy")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    if inputfilename[-5]=="l":
        model = QLearning(1)
        for i in range(500):
            model.estimateQstar(inputfilename)
        writePolicy(model.Q,outputfilename)
    elif inputfilename[-5]=="m":
        model = QLearning(2)
        for i in range(500):
            model.estimateQstar(inputfilename)
        writePolicy(model.Q,outputfilename)
    elif inputfilename[-5]=="e":
        model = QLearning(3)
        for i in range(2000):
            model.estimateQstar(inputfilename)
        writePolicy(model.Q,outputfilename)
    

if __name__ == '__main__':
    main()