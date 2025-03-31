size = int(input("Enter USB size (GB): "))

size *= 2**30

gif = 800 * 600 // 5
jpeg = 800 * 600 * 3 // 25
png = 800 * 600 * 3 // 8
tiff = 800 * 600 * 6

print(format(size // gif, "6") + " image(s) in GIF format can be stored")
print(format(size // jpeg, "6") + " image(s) in JPEG format can be stored")
print(format(size // png, "6") + " image(s) in PNG format can be stored")
print(format(size // tiff, "6") + " image(s) in TIFF format can be stored")
