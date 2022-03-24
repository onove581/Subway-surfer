import cv2
import pyautogui
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

# Khởi tạo mediapipe pose class
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Khởi tạo mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils 

# Hàm phát hiện tư thế Pose của người nổi bật nhất trong ảnh
def detectPose(image, pose, draw=False, display=False):

    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Check if any landmarks are detected and are specified to be drawn.
    if results.pose_landmarks and draw:
    
        # Draw Pose Landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    # Otherwise
    else:

        # Return the output image and the results of pose landmarks detection.
        return output_image, results


def checkHandsJoined(image, results, draw=False, display=False):

    height, width, _ = image.shape
    
    output_image = image.copy()
    
    # Lấy tọa độ Landmark của cổ tay bên Tái
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    # Lấy tọa độ Landmark của cổ tay Phải
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    # Tính khoảng cách giữa 2 điểm landmark Wrist (cổ tay) Left & Right
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    
    # So sánh khoảng cách giữa 2 cổ tay vối số 130
    if euclidean_distance < 130:
        
        # Nếu khoảng cách 2 cổ tay < 130 thì có nghĩa là 2 bàn tay đang nắm lại với nhau
        hand_status = 'Hands Joined'
        
        # Green
        color = (0, 255, 0)
        
    else:
        
        # Ngược lại thì 2 bàn tay chưa nắm lại
        hand_status = 'Hands Not Joined'
        
        # Red
        color = (0, 0, 255)
        
    
    if draw:

        # Write the classified hands status on the image. 
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        # Write the the distance between the wrists on the image. 
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        # Return the output image and the classified hands status indicating whether the hands are joined or not.
        return output_image, hand_status


def checkLeftRight(image, results, draw=False, display=False):

    # Biến này được dùng để lưu vị trí của User (left, right, center).
    horizontal_position = None
    
    height, width, _ = image.shape
    
    output_image = image.copy()
    
    # Lấy tọa độ x của landmark vai Trái
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

    # Lấy tọa x của landmark vai Phải
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

    # Kiểm tra nếu x của cả 2 vai đều < tọa độ x của điểm giữa camera thì User đang ở bên Trái
    if (right_x <= width//2 and left_x <= width//2):
        
        horizontal_position = 'Left'

    # Kiểm tra nếu x của cả 2 vai đều > tọa độ x của điểm giữa camera thì User đang ở bên Phải
    elif (right_x >= width//2 and left_x >= width//2):
        
        horizontal_position = 'Right'
    
    # Kiểm tra nếu x của vai Phải > x của điểm giữa && x vai Trái < x của điểm giữa thì User đang ở Giữa
    elif (right_x >= width//2 and left_x <= width//2):
        
        horizontal_position = 'Center'
        
    if draw:

        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Vẽ 1 đường dọc ở giữa khung hình
        cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
    # Check if the output image is specified to be displayed.
    if display:

        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    else:
    
        return output_image, horizontal_position


def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):

    height, width, _ = image.shape
    
    # Create a copy of the input image to write the posture label on.
    output_image = image.copy()
    

    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)


    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)


    # Tính lại giá trị y hiện tại của điểm giữa 2 vai
    actual_mid_y = abs(right_y + left_y) // 2
    
    lower_bound = MID_Y-15
    upper_bound = MID_Y+100

    # Kiểm tra nếu giá trị y giữa 2 vai hiện tại < giá trị y giữa 2 vai lúc User standing
    if (actual_mid_y < lower_bound):
        
        # Gán giá trị 'Jumping' vào biến
        posture = 'Jumping'
    

    # Kiểm tra nếu giá trị y giữa 2 vai hiện tại > giá trị y giữa 2 vai lúc User standing
    elif (actual_mid_y > upper_bound):
        
        # Gán giá trị 'Crouching' vào biến
        posture = 'Crouching'
    
    else:
        
        # Gán giá trị 'Standing' vào biến
        posture = 'Standing'
        
    # Check if the posture and a horizontal line at the threshold is specified to be drawn.
    if draw:

        
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        
        # Vẽ 1 đường ngang trong khung hình, đi qua vị trí y giữa 2 vai
        cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
        
    
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
    
        
        return output_image, posture        


# Khởi tạo VideoCapture object để đọc hình ảnh từ webcam.
camera_video = cv2.VideoCapture(0)

# Create named window for resizing purposes.
cv2.namedWindow('Game Subway Surfers', cv2.WINDOW_NORMAL)

# Biến này để xác định trạng thái bắt đầu chơi
game_started = False   

# Biến này để lưu vị trí hiện tại của user, 
# 0: user ở left
# 1: user ở center
# 2: user ở right
x_pos_index = 1

# Biến này để xác định
# 0: crouch
# 1: standing
# 2: jump
y_pos_index = 1

# Biến lưu tọa độ y giữa 2 vai
MID_Y = None


# Dùng đếm frames sau khi chập 2 tay lại
counter = 0

# số frames giới hạn để check sau khi chập 2 tay lại
num_of_frames = 10

# Until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    
    frame = cv2.flip(frame, 1)
    
    # Lấy chiều Cao, chiều rộng của frame trong video
    frame_height, frame_width, _ = frame.shape
    
    # Perform the pose detection on the frame.
    frame, results = detectPose(frame, pose_video, draw=game_started)
    
    
    if results.pose_landmarks:
        
        # Kiểm tra nếu game đã bắt đầu chơi
        if game_started:
            # phát hiện vị trí của user ở left or right
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)
            
            # Kiểm tra nếu user di chuyển đến vị trí Left từ Center hoặc đến vị trí center từ right
            if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                
                # Press the left arrow key.
                pyautogui.press('left')
                
                #Cập nhật vị trí hiện tại của User
                x_pos_index -= 1               

            # Kiểm tra nếu User di chuyển đến vị trí Right từ Center hoặc đến vị trí Center từ Left
            elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                
                pyautogui.press('right')
                
                #Cập nhật vị trí hiện tại của User
                x_pos_index += 1

        # Nếu trong trường hợp chưa joined 2 bàn tay lại
        else:
            cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)
        
        
        # Kiểm tra nếu 2 bàn tay đã joined lại
        if checkHandsJoined(frame, results)[1] == 'Hands Joined':

            # Tăng biến đếm frames
            counter += 1
           
            if counter == num_of_frames:
                
                # Nếu game chưa bắt đầu
                if not(game_started):

                    game_started = True

                    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

                    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)

                    # tính điểm tọa độ y giữa 2 vai
                    MID_Y = abs(right_y + left_y) // 2

                    # Click để bắt đầu game
                    pyautogui.click(x=1300, y=800, button='left')

                else:

                    # Press the space key.
                    pyautogui.press('space')

                counter = 0
        else:
            counter = 0
            

        # Kiểm tra nếu tọa độ y ở giữa 2 vai có giá trị thì
        if MID_Y:
            
            # Kiểm tra trạng thái nhảy lên (Jump) hoặc cúi xuống (Crouch)
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
            
            # Kiểm tra nếu User nhảy lên
            if posture == 'Jumping' and y_pos_index == 1:
                pyautogui.press('up')
                # cập nhật lại trạng thái của User
                y_pos_index += 1 

            # Kiểm tra nếu User cúi xuống
            elif posture == 'Crouching' and y_pos_index == 1:
                pyautogui.press('down')
                y_pos_index -= 1
            elif posture == 'Standing' and y_pos_index   != 1:
                y_pos_index = 1
        

    # Otherwise if the pose landmarks in the frame are not detected.       
    else:
        counter = 0
    
    # Display the frame.            
    cv2.imshow('Game Subway Surfers', frame)
    
    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF    
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.                  
camera_video.release()
cv2.destroyAllWindows()