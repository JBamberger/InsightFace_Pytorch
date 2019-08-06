
min_face_size = 20.0

width = 100
height = 1000
min_length = min(height, width)

min_detection_size = 12
factor = 0.707  # sqrt(0.5)

scales = []  # scales for scaling the image
print(min_length)
# scales the image so that
# minimum size that we can detect equals to
# minimum face size that we want to detect
m = min_detection_size/min_face_size
min_length *= m

factor_count = 0
while min_length > min_detection_size:
    print(min_length)
    scales.append(m*factor**factor_count)
    min_length *= factor
    factor_count += 1


print(scales)