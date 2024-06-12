import sys
from pathlib import Path
import gc

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QMainWindow, QStatusBar
from PyQt5.QtGui import QPixmap

from pix2pix import sketch_to_image
from triposr_3d import TripoSR
from music_generation import MusicGenerator


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.main_widget = FormWidget(self)
        self.setCentralWidget(self.main_widget)
        self.init_UI()
    
    def init_UI(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusBar().showMessage('Ready')
        self.setGeometry(200, 100, 400, 500)
        self.setWindowTitle('SH2GH')


class FormWidget(QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        """
        윈도우의 UI를 초기화합니다.
        """
        layout = QVBoxLayout()

        # 파일 업로드 라벨
        self.sketch_label = QLabel('Upload Sketch:')
        layout.addWidget(self.sketch_label)
        
        # 파일 업로드 버튼
        self.upload_button = QPushButton('Upload Sketch')
        self.upload_button.clicked.connect(self.upload_sketch)
        layout.addWidget(self.upload_button)

        # 이미지 키워드 입력 라벨
        self.keyword_label = QLabel('Enter Keywords:')
        layout.addWidget(self.keyword_label)

        # 이미지 키워드 입력창        
        self.keyword_input = QLineEdit(self)
        layout.addWidget(self.keyword_input)

        # 음악 키워드 입력 라벨
        self.music_keyword_label = QLabel('Enter Text for Music Generation:')
        layout.addWidget(self.music_keyword_label)

        # 음악 키워드 입력창
        self.music_keyword_input = QLineEdit(self)
        layout.addWidget(self.music_keyword_input)
        
        # 생성 버튼
        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.generate_content)
        layout.addWidget(self.generate_button)
        
        # 생성된 이미지 라벨
        self.image_label = QLabel('Generated Image:')
        layout.addWidget(self.image_label)
        
        # 생성된 이미지 표시 영역
        self.result_image = QLabel()
        layout.addWidget(self.result_image)
        
        # 생성된 3D 모델 경로
        self.model_label = QLabel('3D Model Path:')
        layout.addWidget(self.model_label)
        
        # 생성된 음악 경로
        self.music_label = QLabel('Generated Music Path:')
        layout.addWidget(self.music_label)

        self.setLayout(layout)
        
    def upload_sketch(self):
        """
        손그림 이미지를 업로드합니다.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            self.sketch_path = file_name
            self.sketch_label.setText(f'Sketch: {file_name}')
        
    def generate_content(self):
        """
        업로드 된 이미지를 기반으로 3D 모델과 음악을 생성합니다.
        """
        if not hasattr(self, 'sketch_path'):
            QMessageBox.warning(self, 'Warning', 'Please upload a sketch first!')
            return
        
        image_keyword = self.keyword_input.text()
        if not image_keyword:
            QMessageBox.warning(self, 'Warning', 'Please enter a image keyword!')
            return
        
        music_keyword = self.music_keyword_input.text()
        if not music_keyword:
            QMessageBox.warning(self, 'Warning', 'Please enter a music keyword!')
            return
        
        # Step 1: Sketch to Image
        self.parent.statusBar().showMessage('Processing sketch to image...')
        generated_image = sketch_to_image(self.sketch_path, image_keyword)
        self.display_image("output/sketch_to_image.jpg")
        
        # Step 2: Image to 3D Model
        self.parent.statusBar().showMessage('Processing image to 3D model...')
        triposr = TripoSR()
        model_3d = triposr.image_to_3D("output/sketch_to_image.jpg")
        self.model_label.setText(f'3D Model Path: {model_3d}')
        del triposr
        gc.collect()
        
        # Step 3: Generate music based on keywords
        self.parent.statusBar().showMessage('Generating music...')
        music_generator = MusicGenerator()
        music = music_generator.generate_music(music_keyword)
        self.music_label.setText(f'Generated Music Path: {music}')
        del music_generator
        gc.collect()

        self.parent.statusBar().showMessage('Ready')
        
    def display_image(self, image_path):
        """
        생성된 이미지를 화면에 표시합니다.
        """
        pixmap = QPixmap(image_path)
        self.result_image.setPixmap(pixmap.scaled(200, 200))

if __name__ == '__main__':
    # 윈도우 표시
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())