# Podcast AdBlocker

Podcast AdBlocker is a comprehensive system designed to intercept and remove advertisements from podcast audio streams in real-time. It consists of three components: a backend that processes the audio, a proxy that intercepts the audio stream, and a browser extension that allows users to easily toggle the ad-blocking functionality.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Makefile Usage](#makefile-usage)
    - [Available Targets](#available-targets)
    - [Using the Makefile](#using-the-makefile)
        - [Set Up and Run Everything](#set-up-and-run-everything)
        - [Set Up Individual Components](#set-up-individual-components)
        - [Run Individual Components](#run-individual-components)
        - [Clean Temporary Files](#clean-temporary-files)
        - [Help](#help)
    - [Notes for Windows Users](#notes-for-windows-users)
- [Backend](#backend)
    - [Manual Installation](#manual-installation)
- [Proxy](#proxy)
    - [Manual Installation](#manual-installation-1)
    - [Trusting mitmproxy CA Certificate](#trusting-mitmproxy-ca-certificate)
- [Browser Extension](#browser-extension)
    - [Installation](#installation)
- [Running the System](#running-the-system)
    - [Linux / macOS](#linux--macos)
    - [Windows](#windows)
- [System Requirements](#system-requirements)
- [Important Notes](#important-notes)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Useful Links](#useful-links)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Structure

The project consists of the following main directories:

Root
|- backend
|- extension
|- proxy

- **backend**: Contains the Flask server that handles audio transcription, classification, and editing.
- **proxy**: Uses `mitmproxy` to intercept audio streams and send them to the backend for processing.
- **extension**: A Chrome extension that controls the proxy settings to intercept audio streams.

## Prerequisites

Ensure you have installed the following tools before proceeding:

1. **Python 3.11+**
    - [Download Python](https://www.python.org/downloads/)
    - Ensure Python is added to your system PATH during installation.

2. **Node.js** (for browser extension development)
    - [Download Node.js](https://nodejs.org/)

3. **Chrome Browser** (for testing the extension)
    - [Download Google Chrome](https://www.google.com/chrome/)

4. **mitmproxy** (for audio interception)
    - [Download mitmproxy](https://mitmproxy.org/)

5. **Poetry** (for Python dependency management)
    - Install using the command:
      ```sh
      curl -sSL https://install.python-poetry.org | python3 -
      ```
    - Alternatively, follow the [Poetry Installation Guide](https://python-poetry.org/docs/#installation)

## Makefile Usage

A `Makefile` is provided to make the setup and execution process as streamlined as possible. It automates tasks like dependency installation, backend setup, proxy setup, and running the components of the project.

### Available Targets

- **all**: Set up and run all components.
- **setup-backend**: Install backend dependencies using Poetry.
- **setup-proxy**: Install proxy dependencies using Poetry.
- **setup-mitmproxy**: Provides instructions on how to install the mitmproxy CA certificate.
- **run-backend**: Run the Flask backend server.
- **run-proxy**: Run `mitmproxy` to intercept podcast streams.
- **run-extension**: Ensures that the browser extension is loaded and enabled.
- **run**: Run both the backend and proxy.
- **clean**: Remove Python temporary files and cache.
- **help**: Display available targets.

### Using the Makefile

#### Set Up and Run Everything

To automatically set up and run all components:
```sh
make all
```

#### Set Up Individual Components

You can also set up individual components:
```sh
make setup-Pipeline
make setup-proxy
make setup-mitmproxy
```

#### Run Individual Components

```sh
make run-Pipeline
make run-proxy
make run-extension
```

#### Clean Temporary Files

```sh
make clean
```

#### Help

To see all available options:
```sh
make help
```

### Notes for Windows Users

If you are using Windows, you might need to run some commands from an Administrator Command Prompt for permission reasons. Additionally, Python is commonly referred to as `python` instead of `python3` on Windows, but the Makefile takes care of this by detecting the OS and adjusting accordingly.

## Backend

The backend uses Flask to process audio streams and remove ads. It consists of the following main files:

- **transcriber.py**: Uses Whisper to transcribe audio into text.
- **classifier.py**: Uses a zero-shot classification model to classify segments as advertisements or content.
- **audio_editor.py**: Uses `pydub` to remove advertisement segments.
- **app.py**: Flask server to handle audio processing requests.

### Manual Installation

If you prefer not to use the Makefile, you can set up the backend manually:

1. **Navigate to the `backend` directory**:
   ```sh
   cd Pipeline
   ```

2. **Install dependencies using Poetry**:
   ```sh
   poetry install
   ```

3. **Run the Flask server**:
   ```sh
   poetry run python app.py
   ```
   The server will be available at `http://localhost:5000`.

### Notes

- The backend uses CPU-based inference for transcription and classification. If a GPU is available, modify the model loading code to use it for improved performance.
- Flask runs on port `5000` by default, but you can change the port by modifying `app.py` as needed.

## Proxy

The proxy uses `mitmproxy` to intercept audio streams and send them to the backend for processing.

### Manual Installation

If you prefer not to use the Makefile:

1. **Navigate to the `proxy` directory**:
   ```sh
   cd proxy
   ```

2. **Install dependencies using Poetry**:
   ```sh
   poetry install
   ```

3. **Start `mitmproxy` with the provided script**:
   ```sh
   poetry run mitmdump -s modify_stream.py
   ```

### Trusting mitmproxy CA Certificate

- **macOS/Linux**:
    1. Run `mitmproxy` and note the message prompting you to install the certificate.
    2. Visit `http://mitm.it` in your browser.
    3. Download and install the certificate by following the provided instructions for your OS.

- **Windows**:
    1. Run `mitmproxy` and note the message about installing the certificate.
    2. Visit `http://mitm.it` in your browser.
    3. Download the `.pem` certificate file.
    4. Open it and install it to "Trusted Root Certification Authorities".

For detailed instructions, refer to [mitmproxy documentation](https://docs.mitmproxy.org/stable/).

## Browser Extension

The extension controls the proxy settings and allows users to toggle the ad-blocking functionality on or off.

### Installation

1. **Navigate to the `extension` directory**:
   ```sh
   cd extension
   ```

2. **Load the extension in Chrome**:
    - Open Chrome and go to `chrome://extensions/`.
    - Enable "Developer mode" (toggle the switch in the top-right).
    - Click "Load unpacked" and select the `extension` directory.

3. **Enable the extension using the popup**:
    - Once loaded, click on the extension icon in the toolbar.
    - Use the toggle to enable or disable the ad-blocker functionality.

### Notes

- The extension allows you to enable or disable the proxy, which is required for intercepting audio streams.
- By default, the extension configures the proxy to point to `localhost:8080`, where `mitmproxy` is running.

## Running the System

To run the entire system end-to-end, follow these steps:

### Linux / macOS

1. **Start the backend**:
   ```sh
   cd Pipeline
   poetry run python app.py
   ```

2. **Start the proxy**:
   ```sh
   cd proxy
   poetry run mitmdump -s modify_stream.py
   ```

3. **Load and enable the browser extension**:
    - Open Chrome and navigate to `chrome://extensions/`.
    - Enable "Developer mode" and load the extension.
    - Click on the extension icon to enable the ad-blocker.

### Windows

1. **Start the backend**:
    - Open Command Prompt and navigate to the `backend` directory:
      ```cmd
      cd backend
      ```
    - Run the backend using Poetry:
      ```cmd
      poetry run python app.py
      ```

2. **Start the proxy**:
    - Open another Command Prompt and navigate to the `proxy` directory:
      ```cmd
      cd proxy
      ```
    - Start `mitmproxy` with the script:
      ```cmd
      poetry run mitmdump -s modify_stream.py
      ```

3. **Load and enable the browser extension**:
    - Open Chrome and navigate to `chrome://extensions/`.
    - Enable "Developer mode" and load the extension.
    - Click on the extension icon to enable the ad-blocker.

## System Requirements

- **Backend**:
    - CPU or GPU for inference.
    - Dependencies like `faster-whisper`, `transformers`, `torch`, `pydub`, and `flask` are installed via Poetry.
- **Proxy**:
    - `mitmproxy` for intercepting HTTP requests.
    - `requests` library to handle backend communication.
- **Browser Extension**:
    - Requires Chrome browser.

## Important Notes

- **Mitmproxy Setup**: `mitmproxy` must be trusted by your browser. Follow the detailed setup process to install the certificate correctly.
- **Ports**: Ensure that ports `5000` (backend) and `8080` (`mitmproxy`) are open and available.
- **Operating System Differences**: On Windows, you may need to set up specific permissions or use an administrator Command Prompt to run certain commands.

## Troubleshooting

- **Flask Server Errors**:
    - Ensure all dependencies are installed using Poetry.
    - Check that the correct Python version (`3.11+`) is used.

- **Proxy Errors**:
    - Ensure `mitmproxy` is configured correctly and that the correct CA certificates are installed in your browser.
    - Make sure `mitmproxy` is running on the correct port (`8080`).

- **Chrome Extension Issues**:
    - Make sure "Developer mode" is enabled in Chrome.
    - If the extension fails to enable the proxy, check the Chrome settings or permissions.

- **Windows Specific Issues**:
    - Run Command Prompt as an administrator if you encounter permissions errors.
    - Make sure that any firewalls or antivirus software are configured to allow connections on the required ports (`5000` and `8080`).

## Future Improvements

- **Model Improvements**: The current transcription and classification use CPU-based inference. Modify the code to use GPU if available for improved performance.
- **Extension Enhancements**: Currently, the extension only toggles ad-blocking. Future versions could allow customization of blocked content and automatic updates.

## Useful Links

- [Python Downloads](https://www.python.org/downloads/)
- [Node.js Downloads](https://nodejs.org/)
- [Google Chrome](https://www.google.com/chrome/)
- [mitmproxy Documentation](https://docs.mitmproxy.org/stable/)
- [Poetry Installation Guide](https://python-poetry.org/docs/#installation)

## License

This project is licensed under the MIT License.

## Acknowledgments

- The transcription module is built using `Whisper`.
- Classification is done using the `valhalla/distilbart-mnli-12-1` model from HuggingFace Transformers.
- `pydub` is used for editing audio files.
