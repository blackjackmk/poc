from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from gui import Ui_MainWindow
import sys
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
    
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionWgraj.triggered.connect(self.load_image)
        self.ui.actionRectangles.triggered.connect(self.rectangles)
        self.ui.actionHSV.triggered.connect(self.hsv)
        self.ui.actionYUV.triggered.connect(self.yuv)
        self.ui.actionSzary.triggered.connect(self.szary)
        self.ui.actionHistogram.triggered.connect(self.histogram)
        self.ui.actionBinaryzacja.triggered.connect(self.binaryzacja)
        self.ui.actionHistogram_2.triggered.connect(self.binaryzacja2)
        self.ui.actionOdejmowanie_obrazu.triggered.connect(self.odejmowanie)
        self.ui.actionKraw_d_pionowa.triggered.connect(self.krawedz_pionowa)
        self.ui.actionKraw_d_pozioma.triggered.connect(self.krawedz_poziom)
        self.ui.actionKraw_d_uko_na.triggered.connect(self.krawedz_ukos)
        self.ui.actionLiniowa_dolna.triggered.connect(self.liniowa_dolna)
        self.ui.actionLiniowa_gorna.triggered.connect(self.liniowa_gorna)
        self.ui.actionLiniwa_z_duplikacj.triggered.connect(self.duplikacja)
        self.ui.actionKrawedzie.triggered.connect(self.krawedzie)
        self.ui.actionKrawedzie_pionowe.triggered.connect(self.krawedz_pionowa2)
        self.ui.actionKrawedzie_ukosne.triggered.connect(self.krawedz_ukos2)
        self.ui.actionNieliniowa_min.triggered.connect(self.filtracjaMin)
        self.ui.actionNieliniowa_max.triggered.connect(self.filtracjaMax)
        self.ui.actionNeiliniowa_mediana.triggered.connect(self.filtracjaMedian)

    def load_image(self):
        src = QFileDialog.getOpenFileName(self, 'Open File', '', 'Images (*.png *.xpm *.jpg)')
        self.ui.original_image.setPixmap(QPixmap(src[0]))

    def get_image(self):
        qimage = self.ui.original_image.pixmap().toImage()
        cvimage = qimage.convertToFormat(QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()

        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        image = cv.cvtColor(arr, cv.COLOR_RGBA2RGB)
        return image

    def show_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        return pixmap

    def rectangles(self):
        blank = np.zeros((100, 100, 3), np.uint8)
        cv.rectangle(blank, (0,0), (50,50), (100, 100 ,100), -1) #gray
        cv.rectangle(blank, (50,0), (100,50), (255, 0 ,255), -1) #pink
        cv.rectangle(blank, (0,50), (50,100), (0, 255 ,255), -1) #yellow
        cv.rectangle(blank, (50,50), (100,100), (255, 0 ,0), -1) #blue
        self.ui.edited_image.setPixmap(self.show_image(blank))

    def hsv(self):
        image = self.get_image()
        hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        self.ui.edited_image.setPixmap(self.show_image(hsv_image))

    def yuv(self):
        image = self.get_image()
        yuv_image = cv.cvtColor(image, cv.COLOR_RGB2YUV)
        self.ui.edited_image.setPixmap(self.show_image(yuv_image))

    def szary(self):
        image = self.get_image()
        height, width, _ = image.shape
        for i in range(height):
            for j in range(width):
                red = image[i, j, 2]
                green = image[i, j, 1]
                blue = image[i, j, 0]
                szary2 = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                image[i, j] = [szary2, szary2, szary2]
        self.ui.edited_image.setPixmap(self.show_image(image))

    def histogram(self):
        image = self.get_image()
        hist = cv.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.show()

    def binaryzacja(self):
        image = self.get_image()
        prog = 127
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        for row in range(gray_img.shape[0]):
            for col in range(gray_img.shape[1]):
                pixel = gray_img[row, col]
                if pixel > prog:
                    gray_img[row, col] = 255
                else:
                    gray_img[row, col] = 0
        self.ui.edited_image.setPixmap(self.show_image(gray_img))

    def binaryzacja2(self):
        img = self.get_image()
        szary = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
        prog_max = np.median(szary)
        obraz_binarny = np.zeros_like(img)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                red = img[row, col, 2]
                green = img[row, col, 1]
                blue = img[row, col, 0]
                szary = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                if szary >= prog_max:
                    obraz_binarny[row, col] = 255
                else:
                    obraz_binarny[row, col] = 0
        self.ui.edited_image.setPixmap(self.show_image(obraz_binarny))
    
    def odejmowanie(self):
        img1 = self.get_image()
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        src = QFileDialog.getOpenFileName(self, 'Obraz do odejmowania', '', 'Images (*.png *.xpm *.jpg)')
        img2 = cv.cvtColor(cv.imread(src[0], 1), cv.COLOR_BGR2GRAY)
        for row in range(img1.shape[0]):
            for col in range(img1.shape[1]):
                pixel = img1[row, col] - img2[row, col]
                if pixel < 0:
                    img1[row, col] = 0
                else:
                    img1[row, col] = pixel
        self.ui.edited_image.setPixmap(self.show_image(img1))

    def krawedz_poziom(self):
        img = self.get_image()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for row in range(img.shape[0] - 1):
            for col in range(img.shape[1]):
                pixel = img[row, col] - img[row + 1, col]
                if pixel < 0:
                    img[row, col] = 0
                else:
                    img[row, col] = pixel
        self.ui.edited_image.setPixmap(self.show_image(img))

    def krawedz_pionowa(self):
        img = self.get_image()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for row in range(img.shape[0]):
            for col in range(img.shape[1] - 1):
                pixel = img[row, col] - img[row, col + 1]
                if pixel < 0:
                    img[row, col] = 0
                else:
                    img[row, col] = pixel
        self.ui.edited_image.setPixmap(self.show_image(img))

    def krawedz_ukos(self):
        img = self.get_image()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for row in range(img.shape[0] - 1):
            for col in range(img.shape[1] - 1):
                pixel = img[row, col] - img[row + 1, col + 1]
                if pixel < 0:
                    img[row, col] = 0
                else:
                    img[row, col] = pixel
        self.ui.edited_image.setPixmap(self.show_image(img))

    def liniowa_dolna(self):
        img = self.get_image()
        maska = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]) / 9
        filtered_image = np.zeros_like(img)
        for c in range(3):
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    filtered_image[i, j, c] = np.sum(img[i - 1:i + 2, j - 1:j + 2, c] * maska)
        self.ui.edited_image.setPixmap(self.show_image(filtered_image))
    
    def liniowa_gorna(self):
        img = self.get_image()
        maska = np.array([[0, -1, 0],
                           [-1, 20, -1],
                           [0, -1, 0]]) / 16
        filtered_image = np.zeros_like(img)
        for c in range(3):
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    filtered_image[i, j, c] = np.sum(img[i - 1:i + 2, j - 1:j + 2, c] * maska)
        self.ui.edited_image.setPixmap(self.show_image(filtered_image))
    
    def duplikacja(self):
        img = self.get_image()
        mask = np.array([[0, -1, 0],
                         [-1, 20, -1],
                         [0, -1, 0]]) / 16
        padded_image = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        sharpened_image = np.zeros_like(padded_image)
        for c in range(3):
            for i in range(2, padded_image.shape[0] - 2):
                for j in range(2, padded_image.shape[1] - 2):
                    convolution_sum = np.sum(padded_image[i - 1:i + 2, j - 1:j + 2, c] * mask)
                    sharpened_image[i, j, c] = convolution_sum
        sharpened_image = sharpened_image[1:-1, 1:-1]
        self.ui.edited_image.setPixmap(self.show_image(sharpened_image))

    def krawedzie(self):
        img = self.get_image()
        filter_poziomy = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
        edited = cv.filter2D(img, -1, filter_poziomy)
        self.ui.edited_image.setPixmap(self.show_image(edited))
    def krawedz_pionowa2(self):
        img = self.get_image()
        filter_pionowy = np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]])
        edited = cv.filter2D(img, -1, filter_pionowy)
        self.ui.edited_image.setPixmap(self.show_image(edited))
    def krawedz_ukos2(self):
        img = self.get_image()
        filter_45 = np.array([[0, 1, 2],
                              [-1, 0, 1],
                              [-2, -1, 0]])
        edited = cv.filter2D(img, -1, filter_45)
        self.ui.edited_image.setPixmap(self.show_image(edited))
    
    def filtracjaMin(self):
        img = self.get_image()
        maska = np.zeros((3,3,3), np.uint8)
        padded_image = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        filtered_image = np.zeros_like(img)
        for c in range(3):
            for i in range(2, padded_image.shape[0] - 2):
                for j in range(2, padded_image.shape[1] - 2):
                    maska[:, :, c] = padded_image[i - 1:i + 2, j - 1:j + 2, c]
                    filtered_image[i,j,c] = np.min(maska[:,:,c])
        filtered_image = filtered_image[1:-1, 1:-1]
        self.ui.edited_image.setPixmap(self.show_image(filtered_image))
    def filtracjaMax(self):
        img = self.get_image()
        maska = np.zeros((3,3,3), np.uint8)
        padded_image = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        filtered_image = np.zeros_like(img)
        for c in range(3):
            for i in range(2, padded_image.shape[0] - 2):
                for j in range(2, padded_image.shape[1] - 2):
                    maska[:, :, c] = padded_image[i - 1:i + 2, j - 1:j + 2, c]
                    filtered_image[i,j,c] = np.max(maska[:,:,c])
        filtered_image = filtered_image[1:-1, 1:-1]
        self.ui.edited_image.setPixmap(self.show_image(filtered_image))
    def filtracjaMedian(self):
        img = self.get_image()
        maska = np.zeros((3,3,3), np.uint8)
        padded_image = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        filtered_image = np.zeros_like(img)
        for c in range(3):
            for i in range(2, padded_image.shape[0] - 2):
                for j in range(2, padded_image.shape[1] - 2):
                    maska[:, :, c] = padded_image[i - 1:i + 2, j - 1:j + 2, c]
                    filtered_image[i,j,c] = np.median(maska[:,:,c])
        filtered_image = filtered_image[1:-1, 1:-1]
        self.ui.edited_image.setPixmap(self.show_image(filtered_image))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())