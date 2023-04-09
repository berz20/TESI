# NV Center vectorial magnetometry Analysis

<br />

Git repository for ODMR data analysis done with confocal microscope, applied on NV centers.

<br />

# Contents

- [**How to start**](#contacts)
- [**Brief description**](#course-program)

<br />
<!--------------------------------------------------------------------->

# How to start

In order to make fitting usable, due to the necessity of using lorentzians functions with negative area, the lmfit.models and lmfit.lineshapes files have been modified adding
invlorentzian to lineshapes and the class InvLorentzianModel to models. Surely this is not an elegant neither smart way to do it, but is the only one that has worked stably.

- Search for the files that should be in your python3.10/site-packages/lmfit (on Linux it's sited in ~/.local/lib)
- Either copy the two files for github repository/startup or modify them by adding the modules (Reccomended way)

<br />
<!--------------------------------------------------------------------->

# Brief description

Directories (really messy):

- data
  - Contains data from lab experiences, dir ODMR contains all sets
- docs
  - Abstract and some useful pdfs
- output
  - Contains all txt files filled with the printed output of every run and in dir img all the images created with file src/find_peak_final.py
- presentation
  - Contains all svg files created with Inkscape and slides
- src_EB
  - variables.py : self explanatory
  - functions.py : self explanatory
  - fit_peak_final.py : find peaks, fits with a InvLorentzianModel (with few changes could be implemented for using Gaussians, Lorentzians and Voigt)
  - test_rotative_vectors.py : (only for 8 peaks spectrums) find peaks, fits, plots multiple, plots peaks points and fits with model from hamiltonians.py evaluating possible angle for axis
  - multiple_vector.py : same left graph but 3d plotting of possible vectors
- usb
  - data from lab
  <br />
  <!--------------------------------------------------------------------->
