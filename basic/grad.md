# 自动求导

参考资料：<https://zhuanlan.zhihu.com/p/145353262>

这个梯度计算是基于**反向传播（Backpropagation）**规则实现的，核心思想是利用**链式法则（Chain Rule）**，从最终的标量 `c` 开始，逐步向前计算每个中间变量的梯度。

---

### 1. **计算图分析**
给定 PyTorch 代码：
```python
import torch
x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D)
z = torch.randn(N, D)

a = x * y  # element-wise multiplication
b = a + z
c = torch.sum(b)  # sum all elements
```
其计算图如下：

```
      x --- (*) --- a --- (+) --- b --- (sum) --- c
            /         \          /
      y ---           z --------
```

- `a = x * y`（逐元素相乘，element-wise multiplication）
- `b = a + z`（逐元素相加）
- `c = torch.sum(b)`（求和）

由于 `c` 是一个**标量**，计算其对 `x, y, z` 的梯度时，可以直接应用链式法则。

---

### 2. **反向传播计算**
#### **(1) 计算 `∂c/∂b`**
由于：
\[
c = \sum b
\]
每个 `b[i, j]` 的导数都是 `1`：
\[
\frac{\partial c}{\partial b[i, j]} = 1
\]
所以：
```python
grad_c = 1.0  # c是标量，对自身的梯度是1
grad_b = grad_c * np.ones((N, D))  # ∂c/∂b = 1
```

#### **(2) 计算 `∂c/∂a` 和 `∂c/∂z`**
由于：
\[
b = a + z
\]
对 `a` 和 `z` 求导：
\[
\frac{\partial c}{\partial a} = \frac{\partial c}{\partial b} \cdot \frac{\partial b}{\partial a} = 1 \cdot \mathbf{1} = \mathbf{1}
\]
\[
\frac{\partial c}{\partial z} = \frac{\partial c}{\partial b} \cdot \frac{\partial b}{\partial z} = 1 \cdot \mathbf{1} = \mathbf{1}
\]
所以：
```python
grad_a = grad_b.copy()  # ∂c/∂a = ∂c/∂b
grad_z = grad_b.copy()  # ∂c/∂z = ∂c/∂b
```

#### **(3) 计算 `∂c/∂x` 和 `∂c/∂y`**
由于：
\[
a = x \cdot y
\]
对 `x` 和 `y` 求导：
\[
\frac{\partial c}{\partial x} = \frac{\partial c}{\partial a} \cdot \frac{\partial a}{\partial x} = \mathbf{1} \cdot y = y
\]
\[
\frac{\partial c}{\partial y} = \frac{\partial c}{\partial a} \cdot \frac{\partial a}{\partial y} = \mathbf{1} \cdot x = x
\]
所以：
```python
grad_x = grad_a * y  # ∂c/∂x = (∂c/∂a) * y
grad_y = grad_a * x  # ∂c/∂y = (∂c/∂a) * x
```

---

### 3. **总结**
梯度计算流程：
1. `grad_c = 1.0` （`c` 是标量，对自身梯度为 `1`）
2. `grad_b = grad_c * np.ones((N, D))` （`∂c/∂b = 1`）
3. `grad_a = grad_b.copy()` （`∂c/∂a = ∂c/∂b`）
4. `grad_z = grad_b.copy()` （`∂c/∂z = ∂c/∂b`）
5. `grad_x = grad_a * y` （`∂c/∂x = ∂c/∂a * y`）
6. `grad_y = grad_a * x` （`∂c/∂y = ∂c/∂a * x`）

这一过程就是**手动实现反向传播**，即 PyTorch 中 `c.backward()` 计算的内容。

---

### 4. **PyTorch 自动计算**
可以使用 PyTorch 验证：
```python
import torch

N, D = 3, 3
x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D)
z = torch.randn(N, D)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()  # 计算梯度

print(x.grad)  # 对比 grad_x
print(y * torch.ones((N, D)))  # 应该等于 grad_x
print(y.grad)  # y 没有 requires_grad，梯度不会计算
```
可以看到 `x.grad` 计算结果和 `grad_x` 一致。

---

### 5. **核心思想**
1. **从标量 `c` 反向传播**
2. **利用链式法则**：\( \frac{\partial c}{\partial x} = \frac{\partial c}{\partial b} \cdot \frac{\partial b}{\partial x} \)
3. **逐步求出 `x, y, z` 的梯度**
4. **与 PyTorch `autograd` 计算一致**

这个过程就是 PyTorch `backward()` 自动做的事情，我们手动推导了一遍。