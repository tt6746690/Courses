

a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

def MaxSumDivCon(array):
    """ Divide and conquer method for finding largest sum for contiguous subarrays of positive and negative number array of size n

    @param: Array array: target array
    @rparam: int maxSum: max sum of contiguous subarrays
    """
    n = len(array)

    if (n == 1):
        return array[0]

    m = n / 2
    leftMaxSum = MaxSumDivCon(array[:m]) # here left subarray is smaller than right
    rightMaxSum = MaxSumDivCon(array[m:])

    # find max sum of overlapping subarray starting from the middle
    leftStart = m - 1
    leftPartMaxSum = array[m - 1]
    while (leftStart >= 0):
        leftPartSum = sum(array[leftStart: m])
        if (leftPartSum > leftPartMaxSum):
            leftPartMaxSum = leftPartSum
        leftStart -= 1

    rightEnd = m
    rightPartMaxSum = array[m]
    while (rightEnd <= n):
        rightPartSum = sum(array[m: rightEnd])
        if (rightPartSum > rightPartMaxSum):
            rightPartMaxSum = rightPartSum
        rightEnd += 1


    return max(leftMaxSum, rightMaxSum, (leftPartMaxSum + rightPartMaxSum))


print(MaxSumDivCon(a))
