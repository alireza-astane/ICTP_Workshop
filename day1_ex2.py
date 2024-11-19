def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def print_fibonacci(n):
    for i in range(2,n):
        print(fibonacci(i),end=" ")
    print("\n")



fibonacci_nums = [0,1] 

def fibonacci_dynamic(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n < len(fibonacci_nums):
        return fibonacci_nums[n]
    else:
        fibonacci_nums.append(fibonacci_dynamic(n-1) + fibonacci_dynamic(n-2))
        return fibonacci_nums[n]

def print_dynamic_fibonacci(n):
    for i in range(2,n):
        print(fibonacci_dynamic(i),end=" ")
    print("\n")


print_fibonacci(12)
print_dynamic_fibonacci(12)