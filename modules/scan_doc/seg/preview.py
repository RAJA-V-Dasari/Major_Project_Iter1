from pathlib import Path

import cv2


images = sorted(Path("output/annotated").glob("*.png"))

if not images:
    print("No annotated images found.")
    exit()

index = 0

while True:

    image = cv2.imread(str(images[index]))

    cv2.imshow("Segmentation Preview", image)

    key = cv2.waitKey(0)

    if key in [ord("q"), 27]:
        break

    elif key in [ord("n"), 83]:
        index = min(index + 1, len(images) - 1)

    elif key in [ord("p"), 81]:
        index = max(index - 1, 0)

cv2.destroyAllWindows()