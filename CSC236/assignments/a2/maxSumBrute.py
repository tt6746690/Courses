

a = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

def MaxSumBrute(array):
    """ Brute force method for finding largest sum for contiguous subarrays of positive and negative number array of size n

    @param: Array array: target array
    @rparam: int maxSum: max sum of contiguous subarrays
    """
    # set default max sum
    # true because array consists of both negative and positive numbers
    maxSum = 0
    for i in xrange(0, len(array)):
        startPosition = i
        endPosition = startPosition + 1
        while(endPosition <= len(array)):
            currentSubarraySum = sum(array[startPosition: endPosition])
            if(currentSubarraySum > maxSum):
                maxSum = currentSubarraySum
            endPosition += 1
    return maxSum


print(MaxSumBrute(a))
