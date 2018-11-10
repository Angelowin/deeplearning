# class Date(object):
#     def day(self,x):
#
#
#         month = x[0]
#         date = x[1]
#         num_date = [31,28,31,30,31,30,31,31,30,31,30,31]
#         sum_day = 0
#         for i in range(month-1):
#             sum_day = sum_day+num_date[i]
#         sum = sum_day+date
#         d = sum % 7
#         if d ==0:
#             print(d)
#             return 7
#         else:
#             print(d)
#             return d
#
# dat = Date()
# dat.day([2,4])

# class dele(object):
#     def de(self,x):
#         x  = x.split(" ")
#         da = []
#         for line in x:
#             try:
#                 da.append(line)
#             except:
#                 pass
#         payload_len = da[0]
#         p = da[1][0]
#         if
num = 5
sum = 20
length = [4,42,40,26,46]
nm = input().sp
length = input()
num = nm[0]
sum = nm[1]
max_a = max(length)
resu = []
for i in range(max_a):
    result = 0
    for j in range(num):
        b = length[j]-i
        b = max(0,b)
        result = result+b
    if result== sum:
        resu.append(i)
print(max(resu))












