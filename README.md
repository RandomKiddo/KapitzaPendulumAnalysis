# Kapitza Pendulum Analysis

The following project involves the numerical analysis and manim animation of a Kapitza pendulum system as a part of a theoretical mechanics course final project. Specifically, we use Manim animations and Hopf bifurcation analysis to inspect the stability of the inverted position of the pendulum in the high-driving, low-amplitude regime. This is done with a mix of analytical and numerical methods. 

![GitHub License](https://img.shields.io/github/license/RandomKiddo/KapitzaPendulumAnalysis)


>[!Note]
> Current version is $\beta$ 1.1. The main body of the work is *completed*, but some minor adjustments and additions may be made.

___

## Implementations 

Reading the [Project notebook file](https://github.com/RandomKiddo/KapitzaPendulumAnalysis/blob/master/src/Project.ipynb) provides a comprehensive step-by-step breakdown of the project. Inspecting the various `.py` files shows the different numerical solving methods used to solve this system. 

The project does the following:
* Derives the Lagrangian, equations of motion, and effective potential for the Kapitza pendulum system, while inspecting the high-driving, low-amplitude regime.
* Implements numerical methods to solve the system and check that turning off the driving components recovers the simple pendulum.
* Dynamic widget analysis of the system with varying parameters.
* $U(\phi,t)$ and $U_{\rm eff}(\phi)$ comparisons for high-driving, low-amplitude regime, with separate PDF derivation.
* Manim animations of the system, with normalized state space axis as well.
* Chaos and bifurcation analysis: using Liapunov exponent to judge chaos. Manim animations of two pendula for chaos, with animated Liapunov exponent graph. Hopf bifurcation discrete analysis with checks on stability vs. instability of the vertical position using the bifurcation plot. Further analysis on the low-amplitude area, showing periodic motion but no chaos.

Included in this repository:
* All project files: the project Jupyter notebook, the PDF derivation file, any `.py` numerical files, and the outputted images.
* The Conda environment used for the project, built for M-architecture OS. 
* The presentation file given as a part of this project. 

> [!NOTE]
> Manim animations are hosted in the releases, as they are big, hosted in the `videos.zip` file in each release description. One example is shown below, but in order to get it to display, it has been converted to a gif, which greatly reduces its overall quality. It is recommended to download the zip for higher-fidelity animations. The zip file is [here](https://github.com/RandomKiddo/KapitzaPendulumAnalysis/releases/tag/beta1.1).

Here is the (low quality) gif: <br />
![Example Animation as Low-Quality Gif](src/example.gif)

> [!WARNING]
> Some video viewers ruin the video when playing high-FPS videos. This includes MacOS's QuicktimePlayer. The 150fps movie (like the one shown here, once downloaded) may randomly slow down, causing it to be longer than the expected 30s duration. If this occurs, it is recommended to use the free-to-use [VLC player](https://www.videolan.org/vlc/), as this has been confirmed to work with the high-FPS outputs.

___

<sub>This page was last edited on 12.09.2025.</sub>
