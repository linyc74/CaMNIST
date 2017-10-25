import numpy as np
import tensorflow as tf
import cv2, time, sys, threading, json, os
from view import *
from controller import *
from dnn import * # version 0.3

cam_id = 0
model_1 = 'exp_17_1024_1_model-lambda1e-09.json'
model_2 = 'exp_17_1022_softmax.json'

class Core(object):
    def __init__(self, cam_id=cam_id):
        # Instantiate a controller object.
        # Pass the core object into the controller object,
        # so the controller can call the core.
        self.controller = Controller(core_obj = self)

        # Instantiate a gui object.
        # Pass the controller object into the gui object,
        # so the gui can call the controller, which in turn calls the core
        self.gui = CamnistGUI(controller_obj = self.controller)
        self.gui.show()

        # The mediator is a channel to emit any signal to the gui object.
        # Pass the gui object into the mediator object,
        # so the mediator knows where to emit the signal.
        self.mediator = Mediator(self.gui)

        self.__init__connect_signals()

        # Start the video thread
        self.start_video_thread(cam_id)

    def __init__connect_signals(self):
        """
        Call the mediator to connect signals to the gui.
        These are the signals to be emitted dynamically during runtime.

        Each signal is defined by a unique str signal name.
        """
        signal_names = ['display_topography', 'progress_update']
        self.mediator.connect_signals(signal_names)

    def start_video_thread(self, cam_id):
        # Pass the mediator into the video thread,
        #   so the thread object can talk to the gui.
        self.video_thread = VideoThread(mediator_obj=self.mediator, cam_id=cam_id)
        self.video_thread.start()
        self.prediction_thread = PredictionThread(video_thread=self.video_thread)
        self.prediction_thread.start()

    def stop_video_thread(self):
        self.video_thread.stop()
        self.prediction_thread.stop()

    def close(self):
        'Should be called upon software termination.'
        self.stop_video_thread()

    # Methods called by the controller object

    def snapshot(self):
        if self.video_thread:
            cv2.imwrite('snapshot.jpg', self.video_thread.imgC)

    def toggle_recording(self):
        if self.video_thread:
            self.video_thread.toggle_recording()

    def zoom_in(self):
        if self.video_thread:
            self.video_thread.zoom_in()

    def zoom_out(self):
        if self.video_thread:
            self.video_thread.zoom_out()

    def apply_camera_parameters(self, parameters):

        if self.video_thread:
            self.video_thread.apply_camera_parameters(parameters)



class PredictionThread(threading.Thread):
    """
    This thread object uses neural network to recognize digits
    in the image from the object VideoThread()

    Contains:
        Two NeuralNet() objects:
            One for telling it's a digit or not
            The other for recognizing digit number
    """
    def __init__(self, video_thread, model_1=model_1, model_2=model_2):
        super().__init__()

        self.video_thread = video_thread

        # Get the absolute path of the local folder
        pkg_dir = os.path.dirname(__file__)

        # Instantiate neural network for telling it's a digit or not
        path = os.path.join(pkg_dir, 'parameters\{}'.format(model_1))
        self.nn_1 = NeuralNet()
        self.nn_1.load_model(path)

        # Instantiate neural network for recognizing digit number
        path = os.path.join(pkg_dir, 'parameters\{}'.format(model_2))
        self.nn_2 = NeuralNet()
        self.nn_2.load_model(path)

        self.isStop = False

    def __resize_img(self, img):
        """
        Resizes img to a column vector of (784, 1), value normalized between 0 and 1

        Args:
            img: cv2 color image, dtype=np.uint8

        Returns:
            column_vector: numpy matrix, dtype=np.float32, shape=(784, 1)
        """
        height, width, channels = img.shape

        # Scale the image by the shorter side
        if height < width:
            scale = 28. / height
        else:
            scale = 28. / width

        #    ( ( final image size ) - ( scaled image size ) ) / 2
        tx = ( (       28         ) - ( width  * scale    ) ) / 2
        ty = ( (       28         ) - ( height * scale    ) ) / 2

        mat = np.float32([ [scale, 0    , tx] ,
                           [0    , scale, ty] ])

        # Convert to gray scale
        X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize and center the image to 28 x 28
        X = cv2.warpAffine(X, mat, dsize=(28, 28))
        # Invert gray scale
        X = 255 - X
        # Reshape X as a column vector
        X = X.reshape(784, 1, order='C').astype(np.float32)
        # Normalize data to between 0..1
        min_ = np.min(X)
        max_ = np.max(X)
        if max_ == min_:
            X[:, :] = 0
            column_vector = X
        else:
            column_vector = (X - min_) / (max_ - min_)

        return column_vector

    def run(self):
        video_thread = self.video_thread
        nn_1 = self.nn_1
        nn_2 = self.nn_2
        __resize_img = self.__resize_img

        window_sizes = [40*i for i in range(1, 13)] # 40, 80, ..., 440, 480
        step = 40

        while not self.isStop:
            img = video_thread.get_img()
            height, width, channel = img.shape

            for size in window_sizes:

                v_steps = int((height-size)/step) # number of vertical steps
                h_steps = int((width-size)/step) # number of horizontal steps

                for v in range(v_steps+1):
                    for h in range(h_steps+1):
                        # Get ROI
                        roi = img[v*step:v*step+size, h*step:h*step+size]

                        # Resize image to (784, 1)
                        X = __resize_img(roi)

                        # Predict it's a digit or not
                        pred_1 = nn_1.predict(X)
                        isDigit, prob = pred_1['category'][0], pred_1['probability'][0]

                        if isDigit == 1 and prob > 0.99:
                            pred_2 = nn_2.predict(X)
                            digit, prob = pred_2['category'][0], pred_2['probability'][0]
                            video_thread.draw_digit_mask(top    = v*step,
                                                         left   = h*step,
                                                         width  = size,
                                                         height = size,
                                                         digit  = digit)

    def stop(self):
        'Called to terminate the video thread.'

        # Shut off main loop in self.run()
        self.isStop = True



class VideoThread(threading.Thread):
    """
    This thread object operates the dynamic image acquisition

    Contains:
        A CustomCamera() object which captures images
    """
    def __init__(self, mediator_obj, cam_id):
        super().__init__()

        # The CustomCamera() class instance is a low-level object
        # that belongs to the self video thread object
        self.cam = CustomCamera(cam_id=cam_id)

        # Mediator emits signal to the gui object
        self.mediator = mediator_obj

        self.__init__connect_signals()
        self.__init__parms()

        # Creat the mask for embedding digits
        h, w = self.monitor_height, self.monitor_width
        self.mask = np.ones((h, w), np.uint8)

    def __init__connect_signals(self):
        """
        Call the mediator to connect signals to the gui.
        These are the signals to be emitted dynamically during runtime.

        Each signal is defined by a unique str signal name.
        """
        signal_names = ['display_image', 'recording_starts', 'recording_ends', 'set_info_text']
        self.mediator.connect_signals(signal_names)

    def __init__parms(self):

        # Parameters for image processing
        self.zoom = 1.0
        self.img_height, self.img_width, channels = self.cam.read().shape

        pkg_dir = os.path.dirname(__file__)
        path = os.path.join(pkg_dir, 'parameters/gui.json')
        gui_parms = json.loads(open(path, 'r').read())
        self.monitor_width = gui_parms['monitor_width']
        self.monitor_height = gui_parms['monitor_height']

        self.__set_resize_matrix()

        # Parameters for administrative logic
        self.isRecording = False
        self.isStop = False
        self.isPause = False
        self.t_0 = time.time()
        self.t_1 = time.time()

    def __set_resize_matrix(self):

        # The transformation matrix self.M: Raw image -> Displayed image

        scale = self.zoom

        # To CENTER the scaled image in the output image,
        #   the translation distance tx, ty must
        #     = half of the difference between
        #       the final image size and the scaled image size

        #    ( ( final image size  ) - (    scaled image size  ) ) / 2
        tx = ( (self.monitor_width ) - (self.img_width  * scale) ) / 2
        ty = ( (self.monitor_height) - (self.img_height * scale) ) / 2

        self.M = np.float32([ [scale, 0    , tx] ,
                               [0    , scale, ty] ])

    def __emit_fps_info(self):
        """
        Emits real-time frame-rate info to the gui
        """

        # Calculate frame rate
        self.t_1, self.t_0 = time.time(), self.t_1
        rate = int( 1 / (self.t_1 - self.t_0))

        text = 'Frame rate = {} fps'.format(rate)

        self.mediator.emit_signal( signal_name = 'set_info_text',
                                   arg = text )

    def run(self):
        """
        The main loop that runs in the thread
        """

        while not self.isStop:

            if self.isPause:
                time.sleep(0.1)
                continue

            self.img_raw = self.cam.read()

            self.img_display = cv2.warpAffine(src=self.img_raw,
                                              M=self.M,
                                              dsize=(self.monitor_height, self.monitor_width),
                                              borderValue=(255,255,255))

            # Embed the digit in the displayed image
            self.img_display = cv2.bitwise_and(src1=self.img_display,
                                               src2=self.img_display,
                                               mask=self.mask)

            if self.isRecording:
                self.writer.write(self.img_display)

            self.mediator.emit_signal( signal_name = 'display_image',
                                       arg = self.img_display )
            self.__emit_fps_info()

            # Reset mask every frame (every iteration of the main loop)
            h, w = self.monitor_height, self.monitor_width
            self.mask = np.ones((h, w), np.uint8)

        # Close camera hardware when the image-capturing main loop is done.
        self.cam.close()

    # Public methods called by the high-level core object.

    def stop(self):
        'Called to terminate the video thread.'

        # Stop recording
        if self.isRecording:
            self.isRecording = False
            self.writer.release()

        # Shut off main loop in self.run()
        self.isStop = True

    def toggle_recording(self):
        if not self.isRecording:
            # Define the codec, which is platform specific and can be hard to find
            # Set fourcc = -1 so that can select from the available codec
            fourcc = -1
            # Create VideoWriter object at 30fps
            w, h = self.monitor_width, self.monitor_height
            self.writer = cv2.VideoWriter( 'camnist.avi', fourcc, 30.0, (w, h) )
            self.isRecording = True

            # Change the icon of the gui button
            self.mediator.emit_signal('recording_starts')

        else:
            self.isRecording = False
            self.writer.release()

            # Change the icon of the gui button
            self.mediator.emit_signal('recording_ends')

    def zoom_in(self):
        if self.zoom * 1.01 < 10.0:
            self.zoom = self.zoom * 1.01
            self.__set_resize_matrix()

    def zoom_out(self):
        if self.zoom / 1.01 > 0.8:
            self.zoom = self.zoom / 1.01
            self.__set_resize_matrix()

    def pause(self):
        self.isPause = True

    def resume(self):
        self.isPause = False

    def apply_camera_parameters(self, parameters):

        if self.cam:
            self.cam.apply_camera_parameters(parameters)

    def get_img(self):
        return self.img_display

    def draw_digit_mask(self, top, left, width, height, digit):
        # Draw digit on the mask
        cv2.putText(img=self.mask,
                    text=str(digit),
                    org=(left, top+height),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0,0,0),
                    thickness=3)

        # Draw 4 sides of the rectangle
        self.mask[top, left:(left+width)] = 0
        self.mask[top+height-1, left:(left+width)] = 0
        self.mask[top:(top+height), left] = 0
        self.mask[top:(top+height), left+width-1] = 0



class CustomCamera(object):
    """
    A customized camera API.
    """
    def __init__(self, cam_id):
        """
        One cv2.VideoCapture object is instantiated.
        If not successfully instantiated, then the cv2.VideoCapture object is None.
        """
        self.cam_id = cam_id

        # Instantiate the cv2.VideoCapture object
        self.cam = cv2.VideoCapture(self.cam_id)

        # If not successfully instantiated,
        # then the cv2.VideoCapture object is None
        if not self.cam.isOpened():
            self.cam = None

        self.__init__parameters()
        self.__init__config()

        # Prepare a blank image for the case when the camera is not working
        self.img_blank = cv2.imread('images/blank.tif')

    def __init__parameters(self):
        """
        Load camera parameters
        """
        # Camera parameter IDs are encoded by the API provided by OpenCV
        self.parm_ids = {'width'        : 3   ,
                         'height'       : 4   ,
                         'brightness'   : 10  ,
                         'contrast'     : 11  ,
                         'saturation'   : 12  ,
                         'hue'          : 13  ,
                         'gain'         : 14  ,
                         'exposure'     : 15  ,
                         'white_balance': 17  ,
                         'focus'        : 28  }

        # Load the parameter values stored in the 'parameter' folder

        # Get the directory of the this python file
        pkg_dir = os.path.dirname(__file__)
        # Join the complete path of the parameter file 'cam.json'
        path = os.path.join(pkg_dir, 'parameters/cam.json')
        # Load parameter values from the json file
        self.parm_vals = json.loads(open(path, 'r').read())

    def __init__config(self):
        """
        Configure the camera with the current parameter values
        """
        if not self.cam is None:
            for key in self.parm_ids:
                self.cam.set( self.parm_ids[key], self.parm_vals[key] )

    def read(self):
        """
        Return the properly rotated image.
        If cv2_cam is None than return the blank.tif image.
        """
        if not self.cam is None:
            _, self.img = self.cam.read()
            self.img = np.rot90(self.img, 0)
            # 1 --- Rotates 90 left
            # 3 --- Rotates 90 right

        else:
            # Must insert a time delay to emulate camera harware delay
            # Otherwise the program will crash due to full-speed looping
            time.sleep(0.01)
            self.img = self.img_blank

        return self.img

    def apply_camera_parameters(self, parameters):
        """
        Takes the argument 'parameters' and configure the cv2 camera

        Args:
            parameters: a dictionary of parameters. key = ('width', 'height', ...)
        """

        for key, value in parameters.items():
            self.parm_vals[key] = value

        self.__init__config()

    def close(self):
        """
        Close this custom camera API. One thing to do is to release the cv2 camera.
        """
        if not self.cam is None:
            self.cam.release()
