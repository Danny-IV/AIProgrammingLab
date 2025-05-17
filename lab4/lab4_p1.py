import operator

fruitList = []


def addFruit(fruit, weight):
    """add fruit and weight to fruitList"""
    for item in fruitList:
        if item[0] == fruit:
            item[1] += weight
            return

    # if there is no such fruit in fruitList
    fruitList.append([fruit, weight])


while True:
    fruit = input("Enter a fruit type (q to quit): ")
    if fruit == "q":
        break
    weight = int(input("Enter the weight in kg: "))

    addFruit(fruit, weight)

# sort fruitList alphabetically
fruitList.sort(key=operator.itemgetter(0))

for item in fruitList:
    print(f"{item[0]}, {item[1]}kg.")
