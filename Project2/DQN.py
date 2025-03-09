import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN,self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.f1 = nn.Linear(self.input_dim,256)
        self.f2 = nn.Linear(256,256)
        self.f3 = nn.Linear(256,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=0.02)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self,state):
        x1 = F.relu(self.f1(state))
        x2 = F.relu(self.f2(x1))
        actions = self.f3(x2)

        return actions

def compute(Q_eval,samples_array):

    Q_eval.optimizer.zero_grad()
    batch_size = 50
    batch = np.random.choice(samples_array.shape[0],batch_size)
    batch_index = np.arange(batch_size,dtype=np.int32)

    
    # state_array = np.array([samples_array[batch][0]%500,samples_array[batch][0]//500])
    # new_state_array = np.array([samples_array[batch][3]%500,samples_array[batch][3]//500])
    state = torch.tensor(samples_array[batch,0]).to(Q_eval.device)
    new_state = torch.tensor(samples_array[batch,3]).to(Q_eval.device)
    reward = torch.tensor(samples_array[batch,2]).to(Q_eval.device)


    action = samples_array[batch,1]
    q_eval = Q_eval.forward(state)[batch_index,action]
    q_next = Q_eval.forward(torch.t(new_state))
    
    q_target = reward + 0.95*torch.max(q_next,dim=1)[0] #gamma not defined

    loss = Q_eval.loss(q_target,q_eval).to(Q_eval.device)
    loss.backward()
    Q_eval.optimizer.step()
    return loss

def main():
    inputfile = "./small.csv"
    samples = pd.read_csv(inputfile)
    samples_array = samples.to_numpy()

    state_size = 1  # Adjust based on problem specifications (10x10 grid)
    n_actions = 4

    Q_eval = DQN(state_size, n_actions)

    for epoch in range(500):
        loss = compute(Q_eval, samples_array)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Example: Test Q-values for a sample state
    test_state = torch.tensor([47], dtype=torch.long)
    q_values = Q_eval.forward(F.one_hot(test_state, state_size).float())
    print("Q-values for state 47:", q_values)

if __name__=='__main__':
    main()