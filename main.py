import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def get_landmark(image):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    image.flags.writeable = False
    result = face_mesh.process(image)
    landmarks = result.multi_face_landmarks[0].landmark
    return result, landmarks

def draw_landmarks(image, result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_list=face_landmark,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            '''
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            '''
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    return image
def record(save_to):
    cap = cv2.VideoCapture(0)
    cap.set(10, 100)
    data_arr = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_cp = frame.copy()
        height, width, channels = frame_cp.shape
        try:
            results, landmarks = get_landmark(image=frame_cp)
            annotated_frame = draw_landmarks(frame_cp, results)
            leftx = landmarks[468].x
            lefty = landmarks[468].y
            rightx = landmarks[473].x
            righty = landmarks[473].y
            centerx = landmarks[168].x
            centery = landmarks[168].y
            ldist = [abs(leftx - centerx), abs(lefty - centery)]
            rdist = [abs(rightx - centerx), abs(righty - centery)]
            print("ldist: ", ldist)
            print("rdist: ", rdist)
            data_arr.append([ldist, rdist])
            cv2.imshow("an", annotated_frame)
        except:
            print("irises not in frame")
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    print("saving")
    np.save(save_to, np.array(data_arr))
    cap.release()
def overlay_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(10, 100)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_cp = frame.copy()
        height, width, channels = frame_cp.shape
        try:
            results, landmarks = get_landmark(image=frame_cp)
            annotated_frame = draw_landmarks(frame_cp, results)
            cv2.imshow("an", annotated_frame)
        except:
            print("irises not in frame")
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
def get_coords(image):
    #obsolete method, MediaPipe added support for this function
    lc = [255, 48, 48]
    rc = [48, 255, 48]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(image_rgb == lc, axis=2)
    coordinates = np.argwhere(mask)
    left_coords = np.mean(coordinates, axis=0).astype(int)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.all(image_rgb == rc, axis=2)
    coordinates = np.argwhere(mask)
    right_coords = np.mean(coordinates, axis=0).astype(int)

    return [left_coords[1], left_coords[0]], [right_coords[1], right_coords[0]]

def plot_raw_data(path):
    data = np.load(path)
    data_reshaped = data.transpose(1, 0, 2)

    line1 = data_reshaped[0]
    line2 = data_reshaped[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(line1[:, 0], line1[:, 1], range(len(line1)), label='Line 1', color='fuchsia')
    ax.plot(line2[:, 0], line2[:, 1], range(len(line2)), label='Line 2', color='chartreuse')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Time')
    ax.set_title('3D Plot of Lines with Time')
    ax.legend()
    plt.show()

def plot_raw_data_anim(path):
    data = np.load(path)

    data_reshaped = data.transpose(1, 0, 2)

    line1 = data_reshaped[0]
    line2 = data_reshaped[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line1_plot, = ax.plot([], [], [], label='Line 1', color='fuchsia')
    line2_plot, = ax.plot([], [], [], label='Line 2', color='chartreuse')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Time')
    ax.set_title('Animated 3D Plot of Lines with Time')
    ax.legend()

    frame_counter = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        line1_plot.set_data(line1[:frame, 0], line1[:frame, 1])
        line1_plot.set_3d_properties(range(frame))
        line2_plot.set_data(line2[:frame, 0], line2[:frame, 1])
        line2_plot.set_3d_properties(range(frame))
        ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
        ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
        ax.set_zlim(0, len(line1))
        frame_counter.set_text('Frame: {}'.format(frame))
        return line1_plot, line2_plot, frame_counter

    time = 8900 #(8 seconds)
    animation = FuncAnimation(fig, update, frames=len(line1), interval=8900/len(line1), blit=True)
    animation.save('animated_plot.gif', writer='imagemagick')
    plt.show()


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#overlay_webcam()
#plot_raw_data("my_array.npy")
#record("my_array.npy")
plot_raw_data_anim("my_array.npy")