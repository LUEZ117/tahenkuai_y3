import os
import cv2
import datetime
class Cap():
    WIDTH = 1920
    HEIGHT = 1080
    NUMBER = 1 # 双目摄像头为2
    DEVICE = 1 # 笔记本已有0号摄像头

    def __init__(self):
        # 运行标志
        self.active = True
        self.flag0 = True
        # 启动设备
        print("CAM initializing...")
        self.init_device()
        print("CAM initialized!")

    def init_device(self, debug=False):
        self.CAM = cv2.VideoCapture(self.DEVICE, cv2.CAP_DSHOW)
        self.CAM.set(3, self.WIDTH * self.NUMBER) # 启用双摄像头
        self.CAM.set(4, self.HEIGHT)
        # self.CAM.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
    
    def frame(self):
        if self.NUMBER==0:
            return self.CAM.read()[1]
        else:
            return self.CAM.read()[1][:, :self.WIDTH], self.CAM.read()[1][:, self.WIDTH:] # right
    
    def exit(self):
        self.CAM.release()
        print("CAM closed!")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = Cap()
    files = 0
    folder = "data/test"
    for jpg in os.listdir(folder):
        files += 1

    while True:
        frame = cap.frame()
        cv2.imshow('cap', frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cap.exit()
            break
        elif key == ord("a"):
            # prefix = datetime.datetime.now().strftime('%m%d')
            cv2.imwrite(f"{folder}/{files}.jpg",frame)
            print(f"saved in {files}.jpg")
            files += 1
