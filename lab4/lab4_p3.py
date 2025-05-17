income = int(input("Enter the taxable income in USD: "))

# calculate tax by cases
if income <= 750:
    tax = round(income * 0.02)
elif income <= 2250:
    tax = round(7.5 + (income - 750) * 0.03)
elif income <= 3750:
    tax = round(37.5 + (income - 2250) * 0.06)
elif income <= 5250:
    tax = round(82.5 + (income - 3750) * 0.09)
elif income <= 7000:
    tax = round(income * 0.03 + (income - 5250) * 0.1)
else:
    tax = round(income * 0.04 + (income - 7000) * 0.12)

result = income - tax

# print the result of tax calc
print(f"Tax due: {tax} USD")
print(f"Final income: {result} USD")
