# Assignment 3: Imaging

In this assignment we will explore some operations with images for several reasons:

* to get familiar with manipulating pixel values in various datatypes and the tools for doing these things in Python and NumPy;
* to learn about techniques used in digital photography and cinematography, as well as in computer vision, videoconferencing, and other camera applications; and
* because the same techniques used for photographs and video also apply to computer generated images that we compute as realistic renderings of 3D scenes.

We’ll look at three different kinds of image operations:

* Pointwise transformations apply to every pixel separately.  We will use them to compute good looking 8-bit color images from raw camera sensor data.
* Convolution filters involve computing weighted sums over a local area of the image.  We will use them to simulate out-of-focus blur and to sharpen images.
* Image warping involves moving content around in an image.  We will apply a simple image warp that corrects for the distortion in a wide angle lens.


## Prerequisites

Evidently you have already found the repository where this file and the starter code reside.  Before you can work on this assignment you will need Python and a few Python packages.

### Python
Python 3 >= 3.6, can be installed from [here](https://www.python.org/downloads/).

If you already have Python installed, you can check the version using
```python3 --version```.

Note: If you have multiple versions of python (i.e. both Python 2 and 3) installed, do one of the following:

* Make Python 3 the default - on a Mac/Linux machine this will be something like  
```sudo ln -s /usr/local/bin/python3 /usr/local/bin/python```  
(replace the paths with whatever is returned by `which python3` and `which python`)  
Then replace all the `python3` commands with `python` (also `pip3` with `pip`).
* Make sure you run the scripts and installations with `python3` and `pip3`.

### Terminal tips for beginners (in MacOS)
* In order to display the full (current) directory path, add this line to your `~/.bash_profile`:  
```export PS1='\u@\H:\w$'```  
Then run ```source ~/.bash_profile```.

* I personally recommend using [iTerm2](https://www.iterm2.com/downloads.html) for command line operations in MacOS, given the better appearance and more flexible configurations. You can find the basic tutorial [here](https://www.iterm2.com/documentation.html).

### Installing dependencies
We have included an ```install.sh``` script for quick installation of all the dependencies using pip. You may need to run ```chmod -x install.sh``` first to make it executable. Alternatively, the individual packages can be installed with either pip or conda. Pip comes by default with Python installations (Python 3 >= 3.4), and conda (Miniconda) can be installed from [here](https://docs.conda.io/en/latest/miniconda.html).

Jupyter Notebook:
* With pip: ```pip3 install notebook```
* With conda: ```conda install -c conda-forge notebook```

Pillow (Python imaging library):
* With pip: ```pip3 install Pillow```
* With conda: ```conda install pillow``` (or use [this link](https://anaconda.org/anaconda/pillow))

Numpy:
* With pip: ```pip3 install numpy```
* With conda: ```conda install numpy```

Matplotlib (for visualization):
* With pip: ```pip3 install matplotlib```
* With conda: ```conda install matplotlib```
* (Alternatively) under Linux: ```sudo apt-get install python3-matplotlib```

nbconvert (for converting Jupyter notebooks to other formats):
* With pip: ```pip3 install nbconvert```
* With conda: ```conda install nbconvert```

tifffile:
* With pip: ```pip3 install tifffile```
* With conda: ```conda install tifffile```

### Running Jupyter notebook
In a terminal window run ```jupyter notebook```.  It should automatically open a browser window pointing at the Jupyter application, but if this doesn’t happen you can just browse to the URL that it prints out.  When you are using Jupyter, all the information is stored in the process that is running in this window, so if you close the window or exit the Jupyter server with control-C, you will lose everything unsaved that is stored in the notebook and in any variables that have been defined in the notebook.

This is not often a major problem, since Jupyter auto-saves pretty often.  But you also need to know that the auto-saves are stored in a file in the same directory with the notebook (it’s called `.ipynb_checkpoints`), so you need to run Jupyter again in the same directory to see them.  When you actually hit “Save” in the Jupyter interface, it will update the actual notebook file.

### Imaging tutorial
Once you have Jupyter running in the browser, open `python-imaging-tutorial.ipynb` to get accustomed to the Jupyter notebook and to get a brief intro to image operations using Numpy and Pillow.

### Saving Python scripts from Jupyter notebook
We use `nbconvert` for converting Jupyter notebook files to other formats such as Markdown and Python scripts (see a list of available formats [here](https://nbconvert.readthedocs.io/en/latest/usage.html)). For example, to convert the tutorial notebook into a script, we use

```jupyter nbconvert --to script python-imaging-tutorial.ipynb```

which outputs a `.py` script of the same name.


## Assignment notebooks

### Pointwise Image Operations

One of the simplest ways to transform an image is to apply a function to each pixel.  This kind of pointwise operation is simple to implement, but has a lot of uses.  

The first part of the assignment explores an application of several pointwise operations in digital photography.  You can find the instructions for it in `Photography.ipynb`.

### Image Warping

Pointwise image operations change the colors and tones in images but the image content stays at the same location in the image: each pixel depends only on the pixel at the same location.  Image warping is in some ways the opposite: the content in the image stays the same color, but it moves from place to place.

In the second part of the assignment we look at some simple image warps and an application to correcting the distortion of wide-angle lenses.  We also explore the effects of different interpolation methods.  You can find the instructions for it in `Distortion.ipynb`.

### Image Filtering

In both of the previous parts, each output pixel depends on a single point in the input: for pointwise operations it's exactly the corresponding pixel, whereas for image warping the source locations come from a more arbitrary mapping.  For other kinds of operations we need output pixels to depend on a *neighborhood* in the input, and convolution is the canonical example.

The final part of the assignment is about convolution with arbitrary 2D filters and with separable 2D filters, and looks at an application to image sharpening.  You can find the instructions for it in `Filtering.ipynb`.


## Creative portion

For the creative portion of this assignment there are two avenues

* If your native enthusiasm is for making wacky images, then think up a cool image warping effect and implement it.  (Twisting?  Bulging?  Kaleidoscopes?)  Or code up a filter that creates a nifty effect when applied to an image.  (Starbursts? Lens Bokeh?  Motion blur?)  You just need to compute your images using raw NumPy in the same way as in the assignment.

* If your native enthusiasm is for making things that go fast, then figure out how to implement all the functions in the convolution and warping notebooks so that they run much faster than the basic versions.  Try to meet these performance goals (where *T* is the time taken by the example box filtering code with r = 7 applied to the full-size test image):
  - 1.6*T* for `convolve_sep(im, gen_gaussian_filter(100, 16.0))` on the full size test image
  - 0.02*T* for `undistort_image(distorted_im, k1, k2, M, 1)` (i.e. with bilinear interpolation)
  - 0.55*T* for the last cell of the Distortion notebook with k = 20 (20 rotation operations with bilinear interpolation)



## What to hand in

We provide a Python module `imaging.py` that contains templates for all the functions that you need to implement, and is the only file (for the basic part) that you will need to hand in on CMS. You can also find the functions to be implemented by searching for "A3TODO" in the notebooks. We recommend implementing the functions in the notebooks first, and test them before copying to the Python script.

For the creative part, submit a 1-page PDF with either:
 * some images you've computed and an explanation of what they are and what code you wrote to make them.
 * some timing test results and an explanation of how you had to change the code to achieve them.
 
A PDF generated from a Jupyter notebook is one appropriate option for this.
```jupyter nbconvert --to pdf [some-notebook].ipynb```
