import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
oupu_size = n_features

#model = nn.Linear(input_size, oupu_size)

class LinearRegression(nn.Module):
    
    def __init__(self,input,output) -> None:
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input,output)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, oupu_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    
    #loss
    l = loss(Y, y_pred)
    
    # gradient = backward pass
    l.backward()
    
    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')
        
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')