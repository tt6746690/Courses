
import math

a = [[0,100], [1,15], [2,5], [3,3], [4,1]]
# n = 5  i:= 0..4

def closestPair(L):
    """ Brute force method for finding the closest pair of points from an array L of bivariate points (x_i, y_i)

    @param: Array[Array] L: input array
    @rparam: Array t: the closest pair
    """

    n = len(L)
    u, v = -1, -1
    min = float('inf')

    for i in range(0, n):
        for j in range(i+1, n):
            distance = findDistance(L[i], L[j])
            if distance < min:
                min = distance
                u = i
                v = j
    return (L[u], L[v])

def findDistance(p1, p2):
    return float(math.sqrt( (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2 ))


print(closestPair(a))
