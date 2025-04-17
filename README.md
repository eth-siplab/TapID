# Research repository for TapID

[Manuel Meier](http://northwards.ch/), [Paul Streli](https://www.paulstreli.com), [Andreas Fender](https://www.andreasfender.com/) and [Christian Holz](https://www.christianholz.net)<br/>
[Sensing, Interaction & Perception Lab](https://siplab.org) <br/>
Department of Computer Science, ETH Zürich

This is the research repository for the IEEE VR 2021 Paper: "TapID: Rapid Touch Interaction in Virtual Reality using Wearable Sensing"

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]


![TapID is a wrist-worn device that detects taps on surfaces and identifies the tapping finger, which, combined with tracked hand poses, triggers input in VR. (a) The user is wearing two TapID bands for (b) touch interaction with surface widgets in VR, e.g., for text input, web browsing, or (c) document authoring using familiar front-end apps. (d) Widgets can also be registered to the body itself, using TapID to detect on-body taps and identify the tapping finger, here to rotate an image held in hand.](https://siplab.org/figures/teasers/TapID.jpg)

TapID enables direct touch input for Virtual Reality on physical surfaces to support tasks like typing, browsing, and object manipulation. Our wrist-worn devices detect taps, identify fingers, and relay events to the VR environment.


## Dataset

The dataset consists of IMU data from 18 participants collected using the TapID wristbands. Each data point represents a tap gesture performed with one of five fingers (thumb, index, middle, ring, or little finger) of each hand.

The data can be downloaded from [here](https://drive.google.com/drive/folders/1njXzVS1SN068bZraQMkag-MXow7iOZ7c).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TapID.git
   cd TapID
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the link above and extract it to the `data` directory.

## Usage

You can run the experiments using the command-line interface:

```bash
# Cross-participant experiment (default mode)
python run_experiment.py --mode crossparticipant --sensors 0 1 2 3

# Cross-block experiment
python run_experiment.py --mode crossblock --sensors 0 1 2 3

# Specify output directory and experiment name
python run_experiment.py --mode crossparticipant --sensors 0 1 2 3 --output_dir results --experiment_name my_experiment
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode (`crossparticipant` or `crossblock`) | `crossparticipant` |
| `--sensors` | List of sensors to use | `[0, 1, 2, 3]` |
| `--sessions` | List of sessions to use | `[0, 1, 2, 3, 4, 5]` |
| `--participants` | List of participants to use | `None` (all participants) |
| `--fingers` | List of fingers to classify | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ` |
| `--output_dir` | Directory to save trained models | `saved_models` |
| `--experiment_name` | Name of the experiment for saving results | Auto-generated |
| `--data_dir` | Directory containing the data | `data` |

## Publication reference

Manuel Meier, Paul Streli, Andreas Fender, and Christian Holz. TapID: Rapid Touch Interaction in Virtual Reality using Wearable Sensing. In Proceedings of Virtual Reality Conference 2021 (IEEE VR). DOI:https://doi.org/10.1109/VR50410.2021.00076

### Direct links

* [TapID paper PDF](https://static.siplab.org/papers/vr2021-TapID.pdf)
* [TapID video](https://www.youtube.com/watch?v=cZl_Sn2dhZY)
* [TapID project page](https://siplab.org/projects/TapID)

### BibTeX reference

```
@inproceedings{vr2021-TapID,
  author = {Meier, Manuel and Streli, Paul and Fender, Andreas and Holz, Christian},
  booktitle = {2021 IEEE Virtual Reality and 3D User Interfaces (VR)},
  title = {TapID: Rapid Touch Interaction in Virtual Reality using Wearable Sensing},
  year = {2021},
  volume = {},
  number = {},
  pages = {519-528},
  doi = {https://doi.org/10.1109/VR50410.2021.00076}
}
```

## Abstract

Current Virtual Reality systems typically use cameras to capture user input from controllers or free-hand mid-air interaction. In this paper, we argue that this is a key impediment to productivity scenarios in VR, which require continued interaction over prolonged periods of time-a requirement that controller or free-hand input in mid-air does not satisfy. To address this challenge, we bring rapid touch interaction on surfaces to Virtual Reality-the input modality that users have grown used to on phones and tablets for continued use. We present TapID, a wrist-based inertial sensing system that complements headset-tracked hand poses to trigger input in VR. TapID embeds a pair of inertial sensors in a flexible strap, one at either side of the wrist; from the combination of registered signals, TapID reliably detects surface touch events and, more importantly, identifies the finger used for touch. We evaluated TapID in a series of user studies on event-detection accuracy (F1 = 0.997) and hand-agnostic finger-identification accuracy (within-user: F1 = 0.93; across users: F1 = 0.91 after 10 refinement taps and F1 = 0.87 without refinement) in a seated table scenario. We conclude with a series of applications that complement hand tracking with touch input and that are uniquely enabled by TapID, including UI control, rapid keyboard typing and piano playing, as well as surface gestures.


## Disclaimer

The dataset and code in this repository is for research purposes only. If you plan to use this for commercial purposes, please [contact us](https://siplab.org/contact). If you are interested in a collaboration with us around this topic, please also [contact us](https://siplab.org/contact).

```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY
WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE
QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE
DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR
CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU
FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL
DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT
NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES
SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH
ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
