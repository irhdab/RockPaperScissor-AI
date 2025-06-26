# Building a Rock Paper Scissor AI with Neural Networks

This Repo is aimed at beginers who want to try and build their own bots with ease. So lets begin!



An AI-powered Rock Paper Scissors game using computer vision and neural networks. This project demonstrates how to build a real-time gesture recognition system using TensorFlow and OpenCV.

## Features

- Real-time gesture recognition using computer vision
- Neural network-based classification with DenseNet121
- Cross-platform compatibility (Windows, macOS, Linux)
- Improved error handling and user experience
- Modular code structure for easy customization
- Training visualization with accuracy/loss plots
- Configurable settings via centralized config file
- Comprehensive unit tests and code quality checks

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RockPaperScissor-AI-.git
   cd RockPaperScissor-AI-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:
   ```bash
   pip install -e .
   ```

3. **Install development dependencies (optional)**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

### 1. Data Collection

Collect training images for each gesture:

```bash
# Collect rock images (100 images, numbered 1-100)
python getData.py rock 1 100

# Collect paper images (100 images, numbered 1-100)  
python getData.py paper 1 100

# Collect scissor images (100 images, numbered 1-100)
python getData.py scissor 1 100
```

**Instructions:**
- Position your hand in the capture area (white rectangle)
- Press **SPACE** to capture an image
- Press **Q** to quit early
- Collect at least 50-100 images per gesture for good results

### 2. Training

Train the neural network on your collected data:

```bash
python train.py ./data
```

**What happens:**
- Loads and preprocesses images from `./data/rock`, `./data/paper`, `./data/scissor`
- Applies data augmentation (flipping, cropping)
- Trains DenseNet121 model with transfer learning
- Saves model as `model.h5` and `model.json`
- Generates training history plots

### 3. Play the Game

Start playing against the AI:

```bash
python play.py
```

**Game Flow:**
1. Press **SPACE** to start
2. Wait for countdown (3, 2, 1)
3. Show your gesture in the capture area
4. See the AI's prediction and response
5. Play 3 rounds and see who wins!

## Testing and Code Quality

### Running Tests

The project includes comprehensive unit tests to ensure code quality:

```bash
# Run all tests
python run_tests.py

# Run specific test modules
python -m pytest tests/test_config.py
python -m pytest tests/test_game_logic.py
python -m pytest tests/test_data_processing.py

# Run tests with coverage
python -m pytest --cov=. tests/
```

### Code Quality Checks

```bash
# Run pylint for code analysis
pylint *.py tests/*.py

# Run flake8 for style checking
flake8 *.py tests/*.py

# Format code with black
black *.py tests/*.py

# Type checking with mypy
mypy *.py
```

### Test Coverage

The test suite covers:
- Configuration validation and settings
- Game logic and scoring rules
- Data processing and image validation
- Model architecture and training functions
- Error handling and edge cases

## Project Structure

```
RockPaperScissor-AI-/
├── getData.py          # Data collection script
├── train.py           # Model training script  
├── play.py            # Game interface
├── config.py          # Configuration settings
├── requirements.txt   # Python dependencies
├── requirements-dev.txt # Development dependencies
├── setup.py          # Installation script
├── run_tests.py      # Test runner
├── .pylintrc         # Pylint configuration
├── README.md         # This file
├── tests/            # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_game_logic.py
│   └── test_data_processing.py
└── data/             # Training data (created by user)
    ├── rock/         # Rock gesture images
    ├── paper/        # Paper gesture images
    └── scissor/      # Scissor gesture images
```

## Configuration

All settings are centralized in `config.py`:

- Data Collection: Capture area, image size, supported formats
- Model: Architecture, hyperparameters, training settings
- Game: Number of rounds, timing, UI settings
- GPU: Memory management and optimization

## Improvements Made

### 1. Code Quality & Structure
- Fixed cross-platform path handling
- Added comprehensive error handling
- Implemented modular, class-based architecture
- Created centralized configuration system
- Added comprehensive unit tests
- Implemented code quality checks

### 2. AI/ML Enhancements
- Improved model architecture with better regularization
- Enhanced data augmentation pipeline
- Added training visualization and metrics
- Implemented proper model checkpointing

### 3. User Experience
- Better UI with clear instructions
- Improved game flow and timing
- Added welcome screen and results display
- Enhanced error messages and feedback

### 4. Technical Improvements
- Updated dependencies to latest versions
- Removed Google Colab dependencies
- Added proper setup and installation scripts
- Implemented GPU optimization settings
- Added development tools and linting

## Tips for Better Results

### Data Collection
- Consistent lighting: Use the same lighting conditions
- Clean background: Use a plain, uncluttered background
- Varied angles: Capture gestures from slightly different angles
- Quality images: Ensure clear, focused images

### Training
- More data: Collect 100+ images per gesture for better accuracy
- Balanced dataset: Ensure equal number of images per class
- Retrain: If accuracy is low, collect more data and retrain
- GPU usage: Use GPU acceleration for faster training

### Gameplay
- Good lighting: Ensure your hand is well-lit
- Steady hand: Keep your hand steady during prediction
- Clear gestures: Make distinct rock, paper, scissor gestures
- Practice: The more you play, the better the AI gets!

## Troubleshooting

### Common Issues

**Camera not working:**
- Check if another application is using the camera
- Try different camera index: `cv2.VideoCapture(1)`

**Model not loading:**
- Ensure `model.json` and `model.h5` files exist
- Check file permissions and paths

**Poor accuracy:**
- Collect more training data
- Ensure consistent lighting and background
- Retrain the model with better data

**Memory errors:**
- Reduce batch size in `config.py`
- Use smaller image size
- Enable GPU memory growth

**Test failures:**
- Ensure all dependencies are installed
- Check that test data directories exist
- Verify configuration settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/yourusername/RockPaperScissor-AI-.git
cd RockPaperScissor-AI-
pip install -e .
pip install -r requirements-dev.txt
```

### Before Submitting
1. Run all tests: `python run_tests.py`
2. Check code style: `pylint *.py tests/*.py`
3. Ensure all tests pass
4. Update documentation if needed

## In-Game Snapshots
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/79742539-95954000-8320-11ea-8b79-bff883454617.PNG"/></p>
<br>Making Predictions In the Game :<br>
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/79742699-dbea9f00-8320-11ea-87fb-3ba3f8a9a760.PNG"/></p>
<br>Bot Playing and Score Updates<br>
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/79742843-148a7880-8321-11ea-87fb-3ba3f8a9a760.PNG"/></p>
<br>Final Outcome<br>
<p align="center"><img src = "https://user-images.githubusercontent.com/37273226/79742889-2c61fc80-8321-11ea-99ef-b55fbff4911c.PNG"/></p>


## Notes:
This model works with staionary backgrounds against which Model has been trained. We could implement Hand Detection followed by Gesture Classifation to improve performance. This is a very basic model that simply tries to make predictions on the input image without any major preprocessing but is enough if you play in the same physical position. If you're new at this try chaning your Dataset and re-train your model multiple times would really help you gain better insights into the project.
