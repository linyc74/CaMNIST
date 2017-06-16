import numpy as np
import cv2, time, sys, threading, json, os
from view import *
from controller import *
from neural_net import *



class Core(object):
    def __init__(self):
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
        self.start_video_thread()

    def __init__connect_signals(self):
        '''
        Call the mediator to connect signals to the gui.
        These are the signals to be emitted dynamically during runtime.

        Each signal is defined by a unique str signal name.
        '''
        signal_names = ['display_topography', 'progress_update']
        self.mediator.connect_signals(signal_names)

    def start_video_thread(self):
        # Pass the mediator into the video thread,
        #   so the thread object can talk to the gui.
        self.video_thread = VideoThread(mediator_obj = self.mediator)
        self.video_thread.start()

    def stop_video_thread(self):
        self.video_thread.stop()

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



class VideoThread(threading.Thread):
    '''
    This thread object operates the dynamic image acquisition

    Contains:
        A CustomCamera() object which captures images
        A NeuralNet() object which performs digit recognition
    '''
    def __init__(self, mediator_obj):
        super(VideoThread, self).__init__()

        # The CustomCamera() class instance is a low-level object
        # that belongs to the self video thread object
        self.cam = CustomCamera()

        # Mediator emits signal to the gui object
        self.mediator = mediator_obj

        # The NeuralNet() instance that performs digit recognition
        self.neural_net = NeuralNet(struc = [784, 300, 10])
        pkg_dir = os.path.dirname(__file__)
        path = os.path.join(pkg_dir, 'parameters/mnist_neural_net_02_1.json')
        self.neural_net.load(path)

        self.__init__connect_signals()
        self.__init__parms()

    def __init__connect_signals(self):
        '''
        Call the mediator to connect signals to the gui.
        These are the signals to be emitted dynamically during runtime.

        Each signal is defined by a unique str signal name.
        '''
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

        self.set_resize_matrix()



        # Parameters for administrative logic
        self.isRecording = False
        self.isStop = False
        self.isPause = False
        self.t_0 = time.time()
        self.t_1 = time.time()

    def set_resize_matrix(self):

        # The first transformation matrix M1: Raw image -> Displayed image

        scale = self.zoom

        # To CENTER the scaled image in the output image,
        #   the translation distance tx, ty must
        #     = half of the difference between
        #       the final image size and the scaled image size

        #    ( ( final image size  ) - (    scaled image size  ) ) / 2
        tx = ( (self.monitor_width ) - (self.img_width  * scale) ) / 2
        ty = ( (self.monitor_height) - (self.img_height * scale) ) / 2

        self.M1 = np.float32([ [scale, 0    , tx] ,
                               [0    , scale, ty] ])



        # The second transformation matrix M2: Displayed image -> MNIST image (28x28)

        # Scale the image by the shorter side
        if self.monitor_height < self.monitor_width:
            scale = 28. / self.monitor_height
        else:
            scale = 28. / self.monitor_width

        #    ( ( final image size ) - (      scaled image size    ) ) / 2
        tx = ( (       28         ) - (self.monitor_width  * scale) ) / 2
        ty = ( (       28         ) - (self.monitor_height * scale) ) / 2

        self.M2 = np.float32([ [scale, 0    , tx] ,
                               [0    , scale, ty] ])

    def run(self):
        '''
        The main loop that runs in the thread
        '''

        while not self.isStop:

            if self.isPause:
                time.sleep(0.1)
                continue

            self.img_raw = self.cam.read()

            h, w = self.monitor_height, self.monitor_width
            self.img_display = cv2.warpAffine(src=self.img_raw,
                                              M=self.M1,
                                              dsize=(w, h),
                                              borderValue=(255,255,255))

            prediction = self.predict_digit(self.img_display)

            self.mask_digit = self.gen_digit_mask(digit=prediction['category'],
                                                  confidence=prediction['probability'])

            # Embed the digit in the displayed image
            self.img_display = cv2.bitwise_and(src1=self.img_display,
                                               src2=self.img_display,
                                               mask=self.mask_digit)

            if self.isRecording:
                self.writer.write(self.img_display)

            self.mediator.emit_signal( signal_name = 'display_image',
                                       arg = self.img_display )
            self.emit_fps_info()

        # Close camera hardware when the image-capturing main loop is done.
        self.cam.close()

    def predict_digit(self, img):

        # Convert to gray scale
        X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize and center the image to 28 x 28
        # The resizing parameters that defines the matrix self.M2
        #   is specified in self.set_resize_matrix()
        X = cv2.warpAffine(X, self.M2, dsize=(28, 28))

        # Invert gray scale
        X = 255 - X

        # Show the small image in a separate window
        # cv2.imshow('digit', X)
        # cv2.waitKey(1)

        # Reshape X as a 1D vector
        X = np.reshape(X, (784, ))

        X = X.astype(np.float)

        # Normalize data to between 0..1
        X = X / 255

        return self.neural_net.predict_single(X)

    def gen_digit_mask(self, digit, confidence):

        h, w = self.monitor_height, self.monitor_width
        mask = np.zeros((h, w), np.float)

        # Draw digit on the mask
        cv2.putText(img=mask,
                    text=str(digit),
                    org=(0, 470),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3*confidence, # font size scaled by the confidence
                    color=(255,255,255),
                    thickness=3)

        # The mask has white background and the digit is in black
        return 255 - mask.astype(np.uint8)

    def emit_fps_info(self):
        '''
        Emits real-time frame-rate info to the gui
        '''

        # Calculate frame rate
        self.t_1, self.t_0 = time.time(), self.t_1
        rate = int( 1 / (self.t_1 - self.t_0))

        text = 'Frame rate = {} fps'.format(rate)

        self.mediator.emit_signal( signal_name = 'set_info_text',
                                   arg = text )

    # Methods commanded by the high-level core object.

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
            self.set_resize_matrix()

    def zoom_out(self):
        if self.zoom / 1.01 > 0.8:
            self.zoom = self.zoom / 1.01
            self.set_resize_matrix()

    def pause(self):
        self.isPause = True

    def resume(self):
        self.isPause = False

    def apply_camera_parameters(self, parameters):

        if self.cam:
            self.cam.apply_camera_parameters(parameters)



class CustomCamera(object):
    '''
    A customized camera API.
    '''
    def __init__(self):
        '''
        One cv2.VideoCapture object is instantiated.
        If not successfully instantiated, then the cv2.VideoCapture object is None.
        '''
        self.cam_id = 0

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
        '''
        Load camera parameters
        '''

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
        '''
        Configure the camera with the current parameter values
        '''

        if not self.cam is None:
            for key in self.parm_ids:
                self.cam.set( self.parm_ids[key], self.parm_vals[key] )

    def read(self):
        '''
        Return the properly rotated image.
        If cv2_cam is None than return the blank.tif image.
        '''

        if not self.cam is None:
            _, self.img = self.cam.read()
            self.img = np.rot90(self.img, 1)
            # 1 --- Rotates 90 left
            # 3 --- Rotates 90 right

        else:
            # Must insert a time delay to emulate camera harware delay
            # Otherwise the program will crash due to full-speed looping
            time.sleep(0.01)
            self.img = self.img_blank

        return self.img

    def apply_camera_parameters(self, parameters):
        '''
        Takes the argument 'parameters' and configure the cv2 camera

        Args:
            parameters: a dictionary of parameters. key = ('width', 'height', ...)
        '''

        for key, value in parameters.items():
            self.parm_vals[key] = value

        self.__init__config()

    def close(self):
        '''
        Close this custom camera API. One thing to do is to release the cv2 camera.
        '''
        if not self.cam is None:
            self.cam.release()


