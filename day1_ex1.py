

def sum_of_multiples(max):
    sum = 0
    for i in range(max):
        if i % 3 == 0 or i % 5 == 0:
            sum += i
    return sum 


    
print(sum_of_multiples(10))
print(sum_of_multiples(1000))
