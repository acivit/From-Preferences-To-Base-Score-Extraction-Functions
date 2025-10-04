# Base-Score-Extraction-Functions
Code for the paper: From User Preferences to Base Score Extraction Functions in Gradual Argumentation

## What's inside?

A simulator for obtaining the arguments' base scores of any argumentation framework for decision-making (introducing decisions, arguments, attacks and supports). The simulator outputs the base scores, the decisions, and the arguments and decisions final strengths.

This can be found at the file ```show_base_scores.py```.

It also incorporates a test of the influence of a single argument to another one, for the latter part of the article. 

This can be found at the file ```test_multiple_attacks_supports.py```.

## Requirements

To run the code, it is necessary to have a new version of python (tested in 3.10) or docker (tested in 27.5.1)

## Usage 

The first step is to activate the submodule [Quantitative-Bipolar-Argumentation](https://github.com/TimKam/Quantitative-Bipolar-Argumentation) git repository. 

To do so, launch the file:

```source install_qbaf.sh```

If this does not work, please clone that repository in the root of this one.


Then, it is required to choose whether to use a virtual environment or docker.

### Virtual environment

Step 1. Create the environment

```virtualenv -p python3 venv``` or ```python3 -m venv venv```

Step 2. Install the requirements

```pip install -r requirements.txt```

Step 3. Install the Quantitative-Bipolar-Argumentation repo

```
cd Quantitative-Bipolar-Argumentation
pip install -e .
cd ..
```

Step 4. Run the code

``` streamlit run show_base_scores.py ```

### Docker

Launch the docker compose file

```docker-compose up --build``` or ```docker compose up --build``` depending on the Docker version.


### Open the simulator

Open a browser with the following address: ```http://localhost:8501```


## Troubleshooting

It might be possible that the installation of the package Quantitative-Bipolar-Argumentation is done in docker, but when launching the app it does not find it. As a temporary solution, it is recommended to execute the docker bash, following:

1. Find the docker image name by using: ```docker images```
2. Execute the docker bash using: ```sudo docker run -it -p 8501:8501 <name_of_the_image>```
3. Launch the streamlit code by: ```streamlit run show_base_scores.py```
4. Open the browser and paste the following address: ```http://localhost:8501``` 
