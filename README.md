# ViSBERT-Patent Search Script

Welcome to ViSBERT-Patent. This is a standalone, multimodal (text + image) patent retrieval system that runs on a local machine using PyTorch and Hugging Face Transformers. 

Please note that Internet connection is required when using the search script.

## Installation & Setup (First Time Only)

Follow these steps to set up the environment on a new machine.

### 1. Open the "Offline_System" folder using an IDE (Preferably VS Code)

### 2. Open your Terminal in VS Code

### 3. Create and Activate a Virtual Environment
It is highly recommended to run this inside an isolated virtual environment.

To create the environment execute this command in the terminal:
python -m venv .venv

Activate it using this command (Terminal):
.venv\Scripts\activate

(Note: If Windows blocks the activation with an Execution Policy error, run this command first: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)

Activate it using this command (Mac/Linux):
source .venv/bin/activate


### 3. Install Required Libraries
With your virtual environment activated (you should see "(.venv)" before the path in your terminal), install the dependencies using this command:
python -m pip install -r requirements.txt

Upon installing all required libraries, you would need to download the Model_Weights and place the downloaded "Model_Weights.pt" on the same directory as the search_script.py file

Model Weights can be downloaded here: https://drive.google.com/file/d/1STiegPiam_aZPFSvKxY6W9ssQbSqQnsZ/view?usp=sharing


### 4. IMPORTANT: NLTK Stopwords Fix
The system uses the NLTK library to filter out common English words during text preprocessing. To prevent "Resource not found" errors or Windows symlink security blocks, you must manually download the NLTK dictionary into your environment BEFORE running the main script.

Run this exact command in your terminal:
python -c "import nltk; nltk.download('stopwords')"

---

## Running the Engine

Once the setup is complete, you can launch the interactive search engine at any time by simply running in the terminal:

python search_script.py

### Supported Query Types:
1. Text Only: Type a phrase or abstract.
2. Image Only: Provide the local path to a patent diagram (e.g., diagram.png).
3. Text + Image (Manual): Provide both a typed phrase and an image path.
4. Patent Folder (XML + Image): Point the system to a folder containing a raw patent XML and its corresponding PNG diagram to automatically parse and search.

The model will instantly calculate cosine similarities using the model, display the Top 50 retrieved patents alongside their confidence labels and classification codes, and optionally provide an interactive visual preview of the Top 5 retrieved patents.