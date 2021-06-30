import torch

# x = torch.randn(4, 5, 5)
# print(x)
# mx = torch.max(x)
# indice = torch.nonzero(x == mx)
# print(mx)
# print(indice)

a = torch.randn(4, 2, 4, 4)
indices = []
for i in range(a.shape[0]):
    x = a[i]
    print(x)
    mx = torch.max(x)
    indice = torch.nonzero(x == mx)
    print(mx)
    print(indice)
    indices.append(indice)

indices = torch.cat(indices, dim=0)
print('------------------------')

print(indices)

print('------------------------')
indx = torch.linspace(0, 3, 4, dtype=torch.long)
# indx = torch.cat((indx, indices), dim=1)
# print(indx)

# print(a[indx, indices[:, 0], indices[:, 1], indices[:, 2]])

a = torch.zeros(4, 2, 4, 4)
print(a)
print(indices)
a[indx, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
print(a)
