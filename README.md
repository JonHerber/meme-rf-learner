# RL Meme Generator

An AI-powered meme generator that learns your humor preferences through reinforcement learning. It captures your facial reactions to memes and uses them as rewards to train a model that creates personalized meme combinations.

## Overview

This project combines:
- **Reinforcement Learning** - A PPO agent learns which meme template + sound combinations make you laugh
- **Computer Vision** - Real-time facial expression analysis using DeepFace
- **Meme Generation** - Combines images and sounds from curated collections
- **Interactive Training** - Your reactions become the reward signal

## Features

- 🎭 Real-time facial emotion detection via webcam
- 🎵 Audio-visual meme playback with synchronized sounds
- 🤖 PPO-based RL agent (stable-baselines3)
- 📊 Exploration bonuses and repeat penalties for diverse meme selection
- 💾 Model checkpointing and resumable training
- 🖥️ Fullscreen or windowed display with optional webcam overlay
- 🎮 Demo, training, and evaluation modes

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A webcam (or test video file for development)
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd meme_generator

# Install dependencies with uv (creates virtual environment automatically)
uv sync

# Or install with dev dependencies
uv sync --dev
```

### Running Scripts

With uv, you can run scripts directly:

```bash
# Run with uv (automatically uses the virtual environment)
uv run python scripts/train_interactive.py --mode demo

# Or activate the virtual environment manually
source .venv/bin/activate
python scripts/train_interactive.py --mode demo
```

### Dependencies

| Category | Packages |
|----------|----------|
| Web Scraping | `requests`, `beautifulsoup4` |
| Data Loading | `gdown` |
| Media | `Pillow`, `pygame`, `opencv-python` |
| Emotion Detection | `deepface`, `tf-keras` |
| Reinforcement Learning | `gymnasium`, `stable-baselines3`, `torch` |
| Utilities | `tqdm`, `loguru` |

## Project Structure

```
meme_generator/
├── config/                  # Configuration files
├── data/
│   ├── models/              # Saved RL models
│   ├── sounds/              # Sound effects (MP3)
│   └── templates/           # Meme template images
├── output/                  # Training outputs
├── scripts/
│   ├── create_test_video.py # Generate test video for development
│   ├── download_templates.py # Download meme templates
│   ├── scrape_sounds.py     # Scrape sounds from web
│   ├── test_meme.py         # Test meme playback
│   ├── test_webcam.py       # Test webcam capture
│   ├── train_interactive.py # Main training script
│   └── train_rl.py          # Standalone RL training
├── src/
│   ├── data/                # Data management
│   │   ├── drive_loader.py  # Google Drive downloads
│   │   ├── sound_manager.py # Sound collection handling
│   │   └── template_manager.py # Template collection handling
│   ├── meme/                # Meme generation
│   │   ├── composer.py      # Combine templates + sounds
│   │   └── player.py        # Display and playback
│   ├── pipeline/            # Training orchestration
│   │   └── orchestrator.py  # Main training loop
│   ├── rl/                  # Reinforcement learning
│   │   ├── agent.py         # PPO agent wrapper
│   │   └── environment.py   # Gymnasium environment
│   ├── scrapers/            # Web scraping
│   │   ├── download_manager.py
│   │   └── sound_scraper.py
│   └── vision/              # Computer vision
│       ├── emotion_analyzer.py  # DeepFace wrapper
│       ├── reward_calculator.py # Reaction → reward conversion
│       └── webcam_capture.py    # Threaded frame capture
├── requirements.txt
└── README.md
```

## Usage

### Demo Mode (Random Memes)

Try out the meme player without training:

```bash
python scripts/train_interactive.py --mode demo --num-memes 5
```

### Training Mode

Train the model with your facial reactions:

```bash
# Basic training (5 episodes)
python scripts/train_interactive.py --mode train

# Extended training with model saving
python scripts/train_interactive.py --mode train --episodes 20 --model output/model.zip

# Fullscreen mode without webcam overlay
python scripts/train_interactive.py --mode train --fullscreen --no-overlay
```

### Evaluation Mode

Test a trained model:

```bash
python scripts/train_interactive.py --mode eval --model output/model.zip --num-memes 10
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | `demo`, `train`, or `eval` | `demo` |
| `--episodes, -e` | Number of training episodes | `5` |
| `--memes-per-episode` | Memes shown per episode | `10` |
| `--num-memes, -n` | Memes for demo/eval mode | `5` |
| `--model` | Model path to load/save | None |
| `--fullscreen, -f` | Fullscreen display | False |
| `--meme-duration` | Seconds per meme | `5.0` |
| `--baseline-duration` | Baseline capture time | `3.0` |
| `--camera, -c` | Camera device index | `0` |
| `--no-overlay` | Disable webcam PiP | False |
| `--verbose, -v` | Debug logging | False |

### Controls

- **ESC** - Skip current meme
- **Q** - Quit training

## How It Works

### Training Loop

1. **Baseline Capture** - Records your neutral facial expression for 3 seconds
2. **Meme Selection** - RL agent chooses a template + sound combination
3. **Meme Playback** - Displays the meme with audio
4. **Reaction Capture** - Monitors your facial expressions during playback
5. **Reward Calculation** - Computes reward based on amusement delta from baseline
6. **Model Update** - PPO agent learns from the reward signal

### Reward Signal

The reward is computed from your facial expressions:

```
amusement = weighted_sum(happy, surprise) - weighted_sum(angry, sad, disgust, fear)
reward = clip((reaction_amusement - baseline_amusement) * scale, -1, 1)
```

### Exploration

The agent uses several mechanisms to explore:
- **Entropy bonus** in PPO for action diversity
- **Exploration bonus** for trying new template/sound combinations
- **Repeat penalty** for using the same combinations repeatedly

## Data Collection

### Download Meme Templates

```bash
python scripts/download_templates.py
```

### Scrape Sound Effects

```bash
python scripts/scrape_sounds.py
```

## Development

### Testing Webcam

```bash
# Preview webcam feed
python scripts/test_webcam.py --mode preview

# Test baseline capture
python scripts/test_webcam.py --mode baseline

# Full emotion analysis test
python scripts/test_webcam.py --mode full-test
```

### Creating Test Video (WSL2/No Webcam)

```bash
python scripts/create_test_video.py
```

## Technical Details

### RL Environment

- **Observation Space**: One-hot encoded last template + sound selection, plus recent reward history
- **Action Space**: `MultiDiscrete([num_templates, num_sounds])`
- **Reward Range**: `[-1, 1]` (clipped)

### PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 3e-4 | Standard PPO default |
| N Steps | 64 | Smaller for interactive training |
| Batch Size | 32 | |
| Gamma | 0.95 | Lower for immediate rewards |
| Entropy Coef | 0.05 | Higher for exploration |
| Network | MLP (64, 64) | Two hidden layers |

## License

This project is for educational and personal use.

## Acknowledgments

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO implementation
- [DeepFace](https://github.com/serengil/deepface) for facial emotion analysis
- [Pygame](https://www.pygame.org/) for audio/visual playback
