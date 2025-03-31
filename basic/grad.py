import torch

torch.manual_seed(0)

N, D = 3, 4

x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D, requires_grad=True)
z = torch.randn(N, D, requires_grad=True)

print("x:\n", x)

# 前向传播
a = x * y
b = a + z
c = b.sum()

print("c:", c.item())

# 计算自动求导的梯度
c.backward()

# 记录 PyTorch 自动计算的梯度
auto_grad_x = x.grad.clone()
auto_grad_y = y.grad.clone()
auto_grad_z = z.grad.clone()

# 清空梯度，以便手动计算
x.grad.zero_()
y.grad.zero_()
z.grad.zero_()

# 手动计算梯度
grad_c = torch.tensor(1.0)
grad_b = grad_c * torch.ones((N, D))
grad_a = grad_b.clone()
grad_z = grad_b.clone()
grad_x = grad_a * y
grad_y = grad_a * x

print("Hand-calculated grad_x:\n", grad_x)
print("Hand-calculated grad_y:\n", grad_y)
print("Hand-calculated grad_z:\n", grad_z)

# 计算 PyTorch 期望的梯度
x.backward(grad_x)
y.backward(grad_y)
z.backward(grad_z)

print("PyTorch grad_x:\n", x.grad)
print("PyTorch grad_y:\n", y.grad)
print("PyTorch grad_z:\n", z.grad)

# 进行比较
print("grad_x correct:", torch.allclose(auto_grad_x, grad_x))
print("grad_y correct:", torch.allclose(auto_grad_y, grad_y))
print("grad_z correct:", torch.allclose(auto_grad_z, grad_z))
