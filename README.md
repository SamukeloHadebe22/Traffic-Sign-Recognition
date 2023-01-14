# Traffic-Sign-Recognition
A system that will identify road traffic signs alongside the road as the driver commutes.

In this project we are aiming to create an application that will identify and classify road signs as the driver commutes and notify the driver in case 
they miss them to aid in avoiding car accidents caused by distracted driving. For this project we will be using Python and its multiple libraries to 
train a Machine Learning model, create a registration and login form and the graphic user interface.

We will begin with installing Python 3, if you don't already have it installed you can download it at https://www.python.org/, and after that we 
should install the Anaconda data science environment to get access to Jupyter Notebook which we will use to develop the system. If you don't already have it
installed, you can download it at https://www.anaconda.com/products/distribution/.

Once that's done, navigate to the Anaconda command prompt where we will install the following libraries using pip:
	- Matplotlib
	- Keras
	- Opencv
	- PIL
	- Tensorflow
	- SQLite
	- Tkinter
	- Numpy
	- Pandas

We will also require a dataset of traffic signs which we will use to train and test the Machine Learning model. This dataset can be downloaded at 
https://www.kaggle.com/datasets/.

After we have successfully installed Python 3, the Anaconda data science environment and all the required libraries, and downloaded our dataset of 
traffic signs we are ready to begin the actual coding and we will start by training the model using the downloaded dataset of traffic signs.

Here is the code to train the model and test it's accuracy internally:

When we have trained a model with a satisfying accuracy score, it's time to put that model to use and implement it within a graphic user interface,
which will also have a login form and enable the user to test the system with the computer's webcam.

Here is the code for the graphic user interface:

			import numpy as np 
			import cv2 as cv
			import sys
			import sqlite3
			from tensorflow import keras
			from keras.models import load_model
			import os
			import tkinter as tk
			from tkinter import *
			import tkinter.scrolledtext as tkscrolled
			from gtts import gTTS

# GLOBAL ATTRIBUTES
			ALLOW_SIGNUP=True # whether the login window should first appear before prediction, or bypassed
			ALLOW_TERMINAL = False # whether to show terminal for executing SQL queries

			"""this is a testing notebook, so no need to load datasets here"""

			#dictionary to label all traffic signs class.
			classes = { 0:"None",
				    1:'Speed limit (20km/h)',
				    2:'Speed limit (30km/h)', 
				    3:'Speed limit (50km/h)', 
				    4:'Speed limit (60km/h)', 
				    5:'Speed limit (70km/h)', 
				    6:'Speed limit (80km/h)', 
				    7:'End of speed limit (80km/h)', 
				    8:'Speed limit (100km/h)', 
				    9:'Speed limit (120km/h)', 
				    10:'No passing', 
				    11:'No passing veh over 3.5 tons', 
				    12:'Right-of-way at intersection', 
				    13:'Priority road', 
				    14:'Yield', 
				    15:'Stop', 
				    16:'No vehicles', 
				    17:'Veh > 3.5 tons prohibited', 
				    18:'No entry', 
				    19:'General caution', 
				    20:'Dangerous curve left', 
				    21:'Dangerous curve right', 
				    22:'Double curve', 
				    23:'Bumpy road', 
				    24:'Slippery road', 
				    25:'Road narrows on the right', 
				    26:'Road work', 
				    27:'Traffic signals', 
				    28:'Pedestrians', 
				    29:'Children crossing', 
				    30:'Bicycles crossing',
					31:'Beware of ice/snow',
				    32:'Wild animals crossing', 
				    33:'End speed + passing limits', 
				    34:'Turn right ahead', 
				    35:'Turn left ahead', 
				    36:'Ahead only', 
				    37:'Go straight or right', 
				    38:'Go straight or left', 
				    39:'Keep right', 
				    40:'Keep left',
				    41:'Roundabout mandatory', 
				    42:'End of no passing', 
				    43:'End no passing veh > 3.5 tons' }

			nclasses = len(classes)
			target_size=None

			#load the trained model to classify sign
			model_path = "model43_32_regv.h5"
			model = None

			try:
			    model = keras.models.load_model(model_path)
			    # automatically infer the shape of the input images
			    config = model.get_config() 
			    input_shape = config["layers"][0]["config"]["batch_input_shape"]
			    target_size = input_shape[1]
			except Exception as e:
			    print("An error occurred :", e)
			    sys.exit()

			def predict_class(image):
			    """
			    @params
			    image : ndarray
				the image from which to detect a value
			    """
			    images = np.array([image])
			    probabilities = model.predict(images, verbose=0)[0]
			    pred_idx = np.argmax(probabilities)
			    pred_proba = probabilities[pred_idx]
			    return pred_idx, pred_proba

			def splitter_predict_class(image, soft_vote=True):
			    """
			    soft_vote is mode where we return the class with the highest probability
			    The opposite of this is called hard_voting where we look at the most predicted class and return that
			    """
			    # speed up opencv
			    cv.setUseOptimized(True)
			    cv.setNumThreads(4) # system-dependent
			    img = image.copy()
			ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
			    ss.setBaseImage(img)
			    ss.switchToSelectiveSearchFast()

			    rects = ss.process()
			    windows = []
			    # optimize here sort by w,h ; pick after each N
			    for x, y, w, h in rects[:400]:
				startx, starty, endx, endy = x, y, x+w, y+h
				roi = img[starty:endy, startx:endx]
				roi = cv.resize(roi, dsize=(target_size, target_size), interpolation=cv.INTER_AREA)
				windows.append(roi)
			    windows = np.array(windows)
			    windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 3)
			    windows = windows/255.0
			    predictions = model.predict(windows, verbose=0)
			    pred_idx = None
			    pred_proba = None
			    N, _ = predictions.shape

			    if soft_vote: # soft voting
				max_probas = predictions.max(axis=0) # N, 43 -> N
				pred_idx = np.argmax(max_probas)
				pred_proba = max_probas[pred_idx]
			    else: # hard voting
				y_preds = np.argmax(predictions, axis=1)
				y_probas = np.max(predictions, axis=1)
				uniques, counts = np.unique(y_preds, return_counts=True)
				mxidx = np.argmax(counts)
				pred_idx = uniques[mxidx]
				  # get probability of most voted
				indexer = (y_preds==pred_idx)
				probas = y_probas[indexer]
				pred_proba = np.sum(probas)/np.sum(indexer)
			    return pred_idx, pred_proba

			def view_image(image, winname="<DEMO>"):
			    cv.namedWindow(winname, cv.WINDOW_NORMAL) # create resizable window
			    cv.imshow(winname, image)
			    cv.waitKey()

			def predict_from_camera(camera_id=0):
			    """
			    @params
			    camera_id: int or str, default=0
				the source of the video to be used in prediction
				-A value of 0 means webcam
				-A string means a path/directory to a video
			    """
			    print("This is a demo of the prediction window that captures from video")
			    print("Press 'q' to quit.")

			    cap = cv.VideoCapture(camera_id)
			    if not cap.isOpened():
				return
			    # this is a simulation of the prediction process in progress
			    winname= "TRAFFIC SIGN DETECTOR"
			    while True:
				run, frame = cap.read()
				if not run or len(frame)==0:
				    break
				  img = cv.resize(frame, (target_size, target_size))
				imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
				#idx, proba = splitter_predict_class(imgrgb, soft_vote=True)
				idx, proba,  = predict_class(imgrgb)

				pred = classes[idx+1]

				text = "{}".format(pred)


				language = 'en'
				myobj = gTTS(text=text, lang=language, slow=False) 
				myobj.save("sign.mp3")
				os.system("sign.mp3")


				frame = cv.putText(frame, text, (10, 40), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
				cv.namedWindow(winname, cv.WINDOW_NORMAL) # create resizable window
				cv.imshow(winname, frame)
				if cv.waitKey(1) == ord('q'):

				    break
			    cv.destroyAllWindows()

			class DatabaseHandler:
			    def __init__(self, database_name="data.db"):
				self.database_name=database_name
				self.conn = sqlite3.connect(self.database_name)
				self.cursor = self.conn.cursor()
				# create users table
				query = """
					CREATE TABLE IF NOT EXISTS users(
					    id integer PRIMARY KEY,
					    username varchar(25) UNIQUE NOT NULL,
					    password varchar(25) NOT NULL
					)
					"""
				self.execute(query)

			    def encrypt_password(self, password):
				# encrypting algorithm for password goes here
				return password

			    def addUser(self, username, password):
				"""adds user credentials to database"""
				password = self.encrypt_password(password)
				query = """INSERT INTO users(username, password) VALUES("{}", "{}");""".format(username, password)
				self.execute(query)

			    def getUser(self, username):
				"""checks for user with username"""
				query = """SELECT * FROM users WHERE username="{}" """.format(username)
				result = self.cursor.execute(query)
				return result.fetchone()
			    def verifyUser(self, username, password):
				"""
				checks if user with given password matching given username exists
				@params
				username : str
				    the username to check
				password : str
				    the un-encrypted raw password to match with username
				"""
				user =  self.getUser(username)
				if user:
				    db_pass=user[2]
				    return (self.encrypt_password(password)==db_pass)
				else:
				    return False

			    def deleteUser(self, username):
				"""deletes user with username"""
				query = """DELETE FROM users WHERE username="{}" """.format(username)
				self.cursor.execute(query)

			    def execute(self, query):
				# a quick database query executer
				result=self.cursor.execute(query)
				self.conn.commit()
				return result

			dbHandler = DatabaseHandler()
			#dbHandler.addUser("anilmyne", "anil")
			dbHandler.getUser("anilmyne")

			def terminal():
			    dbHandler = DatabaseHandler()
			    print("-"*10, "DATABASE ACCESS TERMIAL", "-"*10)
			    print("Type sql queries")
			    while True:
				query = input(">")
				query = query.strip()
				try:
				    result = dbHandler.execute(query).fetchall()
				    if type(result) == list:
					stdout = "\n".join(list(map(str, result)))
					print(stdout)
				    else:
					if stdout is not None:
					    print(stdout)
				except Exception as e:
				    print("An error occurred :", e)
				print()

			if ALLOW_TERMINAL:
			    # show database access termminal
			    import threading
			    terminal_fn = lambda : terminal()
			    t = threading.Thread(target=terminal_fn, args=())
			    t.start()

			# the GUI
			class GUI():
			    def __init__(self, h=240, w=640, title='South African Traffic Signs'):
				self.window=tk.Tk()
				self.window.geometry("{}x{}".format(w, h))
				self.H=h
				self.W=w
				self.dbHandler = DatabaseHandler()
				self.bgcolor='#364156'
				self.window.configure(background=self.bgcolor)

				# WINDOW CATEGORIES
				self.WINDOW_HELP=0
				self.WINDOW_MANUAL=1
				self.WINDOW_ABOUT=2

				# THE MENU
				# the file menu
				self.menubar = tk.Menu(self.window)
				self.window.config(menu=self.menubar)

				self.file_menu = tk.Menu(self.menubar, tearoff=0)
				# adding menu Items
				self.file_menu.add_command(label='Manual', command=lambda: self.generateChildWindow(self.WINDOW_MANUAL))
				self.file_menu.add_command(label='Exit', command=self.window.destroy)
				self.menubar.add_cascade(label="File",menu=self.file_menu)

				self.help_menu = tk.Menu(self.menubar, tearoff=0)
				self.help_menu.add_command(label="Help", command=lambda: self.generateChildWindow(self.WINDOW_HELP))
				self.help_menu.add_command(label="About", command=lambda: self.generateChildWindow(self.WINDOW_ABOUT))
				self.menubar.add_cascade(label="Help", menu=self.help_menu)

				self.window.columnconfigure(0, weight=1)
				self.window.rowconfigure(0, weight=1)
				field_size=40

				  # THE MAIN BLOCK
				self.mid_block=Frame(self.window, borderwidth=8)
				self.mid_block.grid(row=0, column=0)

				heading_text="Welcome"
				self.heading = tk.Label(self.mid_block, text=heading_text, font="Helvetica 18 bold")
				self.heading.grid(row=0, column=0)

				self.msg_view = tk.Label(self.mid_block, text="", font="Helvetica 1", fg="#f00")
				self.msg_view .grid(row=1, column=0)

				# THE CREDENTIAL BLOCK
				self.credential_frame = Frame(self.mid_block, borderwidth=5)
				self.credential_frame.grid(row=2, column=0)

				self.name_label = tk.Label(self.credential_frame, text="Username :")
				self.name_label.grid(row=0, column=0)
				self.username_area = tk.Entry(self.credential_frame, width=field_size)
				self.username_area.grid(row=0, column=1)

				self.password_label = tk.Label(self.credential_frame, text="Password :")
				self.password_label.grid(row=1, column=0)
				self.password_area = tk.Entry(self.credential_frame, width=field_size)
				self.password_area.grid(row=1, column=1)

				# THE BUTTON BLOCK
				self.button_frame = Frame(self.mid_block, borderwidth=5)
				self.button_frame.grid(row=3, column=0)
				self.login_button = tk.Button(self.button_frame, text="LOGIN", command=self.login)
				self.login_button.grid(row=0, column=0, padx=(0, 10))
				self.signup_button = tk.Button(self.button_frame, text="SIGN UP", command=self.signupwin)
				self.signup_button.grid(row=0, column=2, padx=(10, 0))

				self.window.mainloop()
			   def login(self):
				username = self.username_area.get()
				password = self.password_area.get()
				username = username.strip()
				password = password.strip()
				msg = ""
				if len(username) == 0:
				    msg = "Username is required!"
				elif len(password) == 0:
				    msg = "Password is required!"
				else:
				    user = self.dbHandler.getUser(username)
				    if user:
					dbpass = user[2]
					if dbpass != self.dbHandler.encrypt_password(password):
					    msg = "Invalid password!"
				    else:
					msg = "Username does not exist!"
				if msg:
				    self.msg_view.config(text=msg, font="Helvetica 8", fg="#f00")
				else:
				    self.msg_view.config(text=msg, font="Helvetica 1", fg="#f00")
				    self.window.destroy()
				    predict_from_camera()

			    def  signupwin(self):
				self.signUpWindow()
			    def signup(self):
				username = self.childwin_username_area.get()
				password = self.childwin_password_area.get()
				username = username.strip()
				password = password.strip()
				msg = ""
				if len(username) == 0:
				    msg = "Username is required!"
				elif len(password) == 0:
				    msg = "Password is required!"
				else:
				    user = self.dbHandler.getUser(username)
				    if user:
					msg = "Username already exists!"
				    else:
					try:
					    self.dbHandler.addUser(username, password)
					except:
					    msg = "An error occured!"
				if msg:
				    self.childwin_msg_view.config(text=msg, font="Helvetica 8", fg="#f00")
				else:
				    self.childwin_msg_view.config(text=msg, font="Helvetica 1", fg="#f00")
				    self.childwin.destroy()   

			    def signUpWindow(self):
				"""creates sign up window"""
				self.childwin = tk.Toplevel(self.window)
				self.childwin.wm_title("Sign Up")
				field_size=40

				  # THE CREDENTIAL BLOCK
				self.childwin.columnconfigure(0, weight=1)
				self.childwin.rowconfigure(0, weight=1)
				mid_block=Frame(self.childwin, borderwidth=8)
				mid_block.grid(row=0, column=0)

				heading_text="SignUp"
				heading = tk.Label(mid_block, text=heading_text, font="Helvetica 18 bold")
				heading.grid(row=0, column=0)

				self.childwin_msg_view = tk.Label(mid_block, text="", font="Helvetica 1", fg="#f00")
				self.childwin_msg_view.grid(row=1, column=0)

				credential_frame = Frame(mid_block, borderwidth=5);
				credential_frame.grid(row=2, column=0)

				name_label = tk.Label(credential_frame, text="Username :")
				name_label.grid(row=0, column=0)
				self.childwin_username_area = tk.Entry(credential_frame, width=field_size)
				self.childwin_username_area.grid(row=0, column=1)

				password_label = tk.Label(credential_frame, text="Password :")
				password_label.grid(row=1, column=0)
				self.childwin_password_area = tk.Entry(credential_frame, width=field_size)
				self.childwin_password_area.grid(row=1, column=1)

				register_button = tk.Button(mid_block, text="Sign Up", command=self.signup)
				register_button.grid(row=3, column=0)
			    def generateChildWindow(self, wincat):
				"""
				this creates a blank window with text
				@params
				wincat : int
				    the category of window ; this determines what content to display
				"""
				text = None
				heading = None
				if wincat==self.WINDOW_HELP:
				    heading="""HELP"""
				    text = """For any emergency enquiries, you can reach out to our system administrator Samukelo Hadebe, at 072 863 9007. 



				    FREQUENTLY ASKED QUESTIONS

				    #1 How to install the application? The application is available on GitHub as an open-source application freely available to all for the demo as we aim to improve the system rapidly.

				    #2 How to register to the system? If you don't already have an account you can simply click on "Sign Up" when the welcome window pops up, after signing up, only then can you login and begin using the application.

				    #3 On which devices is the first version accessible? The first version of the system is accessible through Windows and Mac as we aim to improve the image classification before making the application available on Play Store for large scale use when it's optimized.



			With this being the first version of the system, we would really appreciate any constructive feedback that would help us improve the system for further versions, especially from car drivers who would such a system or car insurance companies who would have the system implemented by their insured drivers.

				    """
				elif wincat==self.WINDOW_MANUAL:
				    heading="""USER MANUAL"""
				    text = """STEP-BY-STEP GUIDE

					#1 Open the application

				    Navigate to the application on 
				    your device and open it, 
				    allow it to load and a 
				    welcome screen will pop up.


				    #2 Create an account

				    If you don't already have 
				    an account, click on the 
				    sign up button that appears 
				    on the welcome screen.
				    A registration window will 
				    pop up, fill in your 
				    information and click sign 
				    up to successfully create 
				    an account.


				    #3 Login to your account

				    If you already have an 
				    account, simply fill in 
				    your username and password 
				    on the welcome screen.


				    #4 Mount the device

				    Mount the device on the 
				    car windshield to allow 
				    the application to clearly 
				    identified road signs as 
					you drive.


				    #5 Begin driving

				    You are good to go! 
				    As you drive along the 
				    road the application will 
				    identify road signs and 
				    notify you.


				    """

				elif wincat==self.WINDOW_ABOUT:
				    heading="""ABOUT"""
				    text = """On average, in South Africa a car accident occurs once every ten minutes, with 30.8% of those accidents being fatal. One of the major causes of car accidents is distracted driving, to which we decided to bring about a new application that will assist drivers as they travel in national roads to keep them and other drivers safer and preventing some car accidents, leading to more lives being spared.



			Drivers can simply install the application, sign up and login then the system is ready for use. Mounting the computer device on the car windshield as you would a GPS device is the best way to get the optimal use of the system. Once mounted the application will identified road signs alongside the road and notify the driver of the identified road sign.
				    """
				  self.subwin = tk.Toplevel(self.window)
				self.subwin.wm_title(heading)

				# MAIN SECTION
				mid_block=Frame(self.subwin, borderwidth=8)
				mid_block.grid(row=0, column=0)

				# HEADING SECTION
				heading_label = tk.Label(mid_block, text=heading, font="Helvetica 13 bold")
				heading_label.grid(row=0, column=0, sticky="NSEW")

				text_area = tkscrolled.ScrolledText(mid_block, width=65, height=25, wrap='word')
				text_area.insert(1.0, text)
				text_area["state"] = tk.DISABLED
				text_area.grid(row=1, column=0, rowspan=10)

			if ALLOW_SIGNUP:
			    gui = GUI()
			else:
			    predict_from_camera()

			This is the output generated from the above code:

![111](https://user-images.githubusercontent.com/112726898/212461772-2cab1684-64b4-4dfb-8b7f-b1f7fc583c30.JPG)
![222](https://user-images.githubusercontent.com/112726898/212461776-9b300433-bd49-4cf8-bf4c-45dd13363c9b.JPG)
![555](https://user-images.githubusercontent.com/112726898/212461781-181a1368-acc5-4030-b5f7-3190e405b64c.JPG)
![666](https://user-images.githubusercontent.com/112726898/212461785-cd87d353-9bab-4a61-853c-1b0d303c583f.JPG)
![777](https://user-images.githubusercontent.com/112726898/212461805-567ebc30-b9ac-4daa-9532-b384a767e007.JPG)
![888](https://user-images.githubusercontent.com/112726898/212461820-aee29d45-ff41-4917-b5f9-8256462a4f15.JPG)
![999](https://user-images.githubusercontent.com/112726898/212461826-7d28cfaf-1be1-4616-b9d4-14b4c15e7701.JPG)


