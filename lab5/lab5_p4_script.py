from lab5_p4 import Point
import math

p1 = Point(0, 0)
print(f"p1 = {p1}")

p2 = Point(3, 4)
p3 = Point(0, 1)

print(f"p2 = {p2}")
print(f"p3 = {p3}")
print(f"p2 + p3 = {p2 + p3}")
print(f"p2 - p3 = {p2 - p3}")
print(f"3 * p2 = {3* p2}")
print(f"p2 / 2 = {p2 / 2}")

p2_symmetric_about_p3 = p2.symmetric(p3)
p2_rotated_90deg_about_p3 = p2.rotate(p3, math.pi / 2)

print(f"p2's symmetric point about p3 = {p2_symmetric_about_p3}")
print(f"p2 rotated 90 degress ccw about p3 = {p2_rotated_90deg_about_p3}")
