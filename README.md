# PyDIC-mod

#🔬 Open-Hole Tensile Test Analysis (Modified PyDIC)
This project provides a customized version of the PyDIC library, specifically adapted for educational purposes in mechanical engineering design and experimental mechanics. It focuses on analyzing Stress Concentration around a central hole in a tensile specimen using Digital Image Correlation (DIC).

#✨ Key Modifications
Interactive Hole Masking: A new feature that allows users to manually select the hole area via mouse clicks to exclude noisy data and artifacts.

Stress Concentration Focus: The post-processing logic is optimized to visualize local strain distribution around a circular discontinuity.

Student-Friendly Guide: Updated instructions and error-handling for Windows-based Python environments.

#🛠️ Installation & Setup
To ensure all dependencies are correctly installed, run the following command in your VS Code terminal:

Bash
# For Windows users (Recommended)
py -m pip install numpy scipy matplotlib opencv-python opencv-contrib-python

# For Mac/Linux users
pip install numpy scipy matplotlib opencv-python opencv-contrib-python
📸 Experimental Procedure
Specimen Prep: Apply white primer and black spray speckles.

Recording: Set up a primary high-res camera on a tripod. Use a secondary phone to capture the UTM load display and a timer for synchronization.

Data Extraction: Extract 6 high-quality original frames (not screenshots!) representing different load stages.

Metadata: Create a meta-data.txt in the img/ folder with columns: image_file, load(N), and time(s).

#🚀 How to Run
Place your images and meta-data.txt in the img/ directory.

Run the script:

Bash
py main.py
Interactive Masking: When the first window pops up, click the Top-Left and Bottom-Right corners of the hole noise.

Results: Close all pop-up windows to see the calculated Young's Modulus (E) and Poisson's Ratio (ν) in your terminal.

#📜 Credits & License
This version is a modified fork of PyDIC (Original source: https://gitlab.com/damien.andre/pydic).

Original Author: Damien ANDRE (Limoges, France)
Modified by: [Kyung Yun Choi/Hongik University]

License: Distributed under the GNU General Public License v3.0.
