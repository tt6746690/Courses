
import math

l = 26


odds = 2
dic = {}
print("tuple: (4, 6, 10)")

while(odds < 100):
    for i in xrange(0,l): # 10
        for j in xrange(0,l):  # 6
            for t in xrange(0,l):  # 4
                tup = (t, j, i)
                boo = True if 10*i + 6*j + 4*t == odds else False
                if(boo):
                    if(not dic.has_key(odds)):
                        dic[odds] = []
                    dic[odds].append(tup)
    odds += 2


# for odd in dic:
#     print(str(odd) + " :" + str(len(dic[odd])) + "--" + str(dic[odd]))


for odd in dic:
    if(odd > 12):
        print("T("+str(odd) + ")=" + str(len(dic[odd]))
        + ": (-4) T("+str(odd-4)+")=" +str(len(dic[odd-4]))
        + ": (-6) T(" + str(odd-6) +")=" + str(len(dic[odd-6]))
        + "; (-10) T("+str(odd-10)+")="+str(len(dic[odd-10]))
        + "; (-2) T("+str(odd-8)+")=" +str(len(dic[odd-2]))
        + "; (-8) T("+str(odd-8)+")=" +str(len(dic[odd-8]))

        + " ; T(n-4) + T(n-6) - T(n-10) + floor(n//10)= " + str( len(dic[odd-4])+len(dic[odd-6])-len(dic[odd-10]) + math.floor(odd//10) ) \
        + " and if T(n) = T(n-4) + T(n-6) - T(n-10): " + str(len(dic[odd-4])+len(dic[odd-6])-len(dic[odd-10]) == len(dic[odd])))

# l = 26
#
#
# integers = 0
# dic = {}
# print("tuple: (4, 6, 10)")
#
# while(integers < 100):
#     for i in xrange(0,l): # 10
#         for j in xrange(0,l):  # 6
#             for t in xrange(0,l):  # 4
#                 tup = (t, j, i)
#                 boo = True if 5*i + 3*j + 2*t == integers else False
#                 if(boo):
#                     if(not dic.has_key(integers)):
#                         dic[integers] = []
#                     dic[integers].append(tup)
#     integers += 1
#
#
# # for odd in dic:
# #     print(str(odd) + " :" + str(len(dic[odd])) + "--" + str(dic[odd]))
#
#
# for odd in dic:
#     if(odd > 7):
#         print("T("+str(odd) + ")=" + str(len(dic[odd]))
#         + ": (-2) T("+str(odd-2)+")=" +str(len(dic[odd-2]))
#         + ": (-3) T(" + str(odd-3) +")=" + str(len(dic[odd-3]))
#         + "; (-5) T("+str(odd-5)+")="+str(len(dic[odd-5]))
#         # + "; (-2) T("+str(odd-8)+")=" +str(len(dic[odd-2]))
#         # + "; (-8) T("+str(odd-8)+")=" +str(len(dic[odd-8])))
#
#         + " ; T(n-2) + T(n-3) - T(n-5) + floor(n//5)= " + str( len(dic[odd-2])+len(dic[odd-3])-len(dic[odd-5]) + math.floor(odd//5) ) \
#         + " and if T(n) = T(n-2) + T(n-3) - T(n-5): " + str(len(dic[odd-2])+len(dic[odd-3])-len(dic[odd-5]) == len(dic[odd])))
