# demo scalars
a = 10
b = 6.5
#print(type(a))
#print(type(b))
# print(a+b)
# print(a-b)
# print(a*b)
# print(a/b)

#check is scalar or not
# import numpy as np
a = 10
b = 6.5
# print(np.isscalar(a))
# print(np.isscalar(b))

#demo vectors
# import numpy as np
a = [10,11]
b = [14,15]
# print(a+b)
# print(np.add(a, b))
# print(np.cross(a, b))

# demo matrix
# import numpy as np
# from numpy import matrix
a = matrix([[1,2], [3,4]])
b = matrix([[4, 3], [2,1]])
x = a.mean(0)
y = a.mean(1)
add = np.matmul(a, b)
mul = np.add(a, b)
t1 = np.transpose(a)
t2 = np.transpose(b)

# demo tensor
#import numpy as np
# import torch
a = torch.Tensor([2, 6])
# print(type(a))
# print(a.shape)