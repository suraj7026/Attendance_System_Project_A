import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(rect)
    plt.show()


filename = "india.jpeg"
pixels = plt.imread(filename)
detector = MTCNN()
faces = detector.detect_faces(pixels)
draw_facebox(filename, faces)
