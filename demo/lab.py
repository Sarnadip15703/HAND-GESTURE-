nums = []
for i in range(10):
    nums.append(int(input(" enter no.")))

def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
        
    return True

even_list = []
prime_list = []

even_list = [num for num in nums if num % 2 == 0]
prime_list = [num for num in nums if is_prime(num)]

print("Even numbers are: ", even_list)
print("Prime numbers are: ", prime_list)


# print("all 3 digit armstrong number are :")
# for n in range(100,1000):
#     sum=0
#     num=n
#     while num>0:
#         a=num//100
#         b=(num//10)%10
#         c=num%10
#         sum=sum+(a**3+b**3+c**3)
        
#         if n==sum:
#             print(n)
#         break