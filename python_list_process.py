import random

# minus
print('minus')
a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
b = [2, 5, 7]
c = [item for item in a if item not in b]
print(c)

# add
print('add')
a = [1, 2, 3]
b = [3, 4, 5, 6]
c = a + b
print(c)

# shuffle
print('shuffle')
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(a, 5)
print(slice)

# shuffle and minus
print('shuffle and minus')
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(a, 3)
minus = [item for item in a if item not in slice]
print(slice)
print(minus)

# shuffle the units in the list
print('shuffle the units in the list')
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(a)
print(a)
