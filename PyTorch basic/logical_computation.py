import torch

# normal FloatTensor
a = torch.Tensor((1, 1, 0, 0, 1, 1))
b = torch.Tensor((1, 0, 0, 1, 1, 1))
print(a.dtype)                                      # torch.float32
print(a.shape)                                      # torch.Size([6])

# Less than sign
lt_sign = a < b
print(lt_sign.dtype)                                # torch.uint8
lt_sign = lt_sign.type(torch.FloatTensor)
print(lt_sign.dtype)                                # torch.float32
print(lt_sign.shape)                                # torch.Size([6])
print(lt_sign)                                      # tensor([0., 0., 0., 1., 0., 0.])

# More than sign
mt_sign = a > b
mt_sign = mt_sign.type(torch.FloatTensor)
print(mt_sign)                                      # tensor([0., 1., 0., 0., 0., 0.])

# Equal sign
eq_sign = a == b
eq_sign = eq_sign.type(torch.FloatTensor)
print(eq_sign)                                      # tensor([1., 0., 1., 0., 1., 1.])

# if you want to use logical computation
# convert to torch.ByteTensor
a = a.type(torch.uint8)
b = b.type(torch.uint8)

# logical and
and_sign = a & b
and_sign = and_sign.type(torch.FloatTensor)
print(and_sign)                                     # tensor([1., 0., 0., 0., 1., 1.])

# logical or
or_sign = a | b
or_sign = or_sign.type(torch.FloatTensor)
print(or_sign)                                      # tensor([1., 1., 0., 1., 1., 1.])

# logical xor
xor_sign = a | b
xor_sign = xor_sign.type(torch.FloatTensor)
print(xor_sign)                                     # tensor([1., 1., 0., 1., 1., 1.])

# logical not
not_sign = ~ a
not_sign = not_sign.type(torch.FloatTensor)
print(not_sign)                                     # tensor([0., 0., 1., 1., 0., 0.])
