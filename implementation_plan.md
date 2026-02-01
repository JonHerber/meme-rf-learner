# RL Meme Generator - Implementation Plan

## Project Overview
A Reinforcement Learning pipeline that learns user humor preferences by:
1. Combining meme pictures, sounds, and captions
2. Capturing facial reactions via OpenCV/webcam
3. Using laughter/amusement scores as RL rewards
4. Learning optimal combinations over time

## Project Structure
```
meme_generator/
├── requirements.txt
├── config/
│   └── settings.py
├── src/
│   ├── scrapers/           # Phase 1: Sound scraping
│   │   ├── sound_scraper.py
│   │   └── download_manager.py
│   ├── data/               # Phase 2: Data management
│   │   ├── drive_loader.py
│   │   ├── template_manager.py
│   │   └── sound_manager.py
│   ├── vision/             # Phase 3: Facial reaction
│   │   ├── webcam_capture.py
│   │   ├── emotion_analyzer.py
│   │   └── reward_calculator.py
│   ├── meme/               # Phase 4: Meme generation
│   │   ├── composer.py
│   │   └── player.py
│   ├── rl/                 # Phase 5: RL pipeline
│   │   ├── environment.py
│   │   └── agent.py
│   └── pipeline/           # Phase 6: Integration
│       └── orchestrator.py
├── data/
│   ├── sounds/             # Downloaded MP3 files
│   ├── templates/          # Meme templates
│   └── models/             # Saved RL models
└── scripts/
    ├── scrape_sounds.py
    └── train.py
```

## Implementation Phases

### Phase 1: Sound Scraper ✅ COMPLETE
- 200+ sounds scraped from myinstants.com
- Uses requests + BeautifulSoup (no Selenium needed)
- MP3 URLs extracted directly from button onclick attributes

### Phase 2: Template Management ✅ COMPLETE
- 75 meme templates downloaded from Google Drive
- Uses gdown for public folder downloads
- Recursive scanning for nested directories

### Phase 3: Facial Reaction Scoring ✅ COMPLETE
**Goal**: Capture facial reactions via webcam and compute RL rewards

**Files created**:
- `src/vision/__init__.py` - Module exports
- `src/vision/webcam_capture.py` - Threaded frame capture with ring buffer
- `src/vision/emotion_analyzer.py` - DeepFace wrapper with baseline capture
- `src/vision/reward_calculator.py` - Baseline comparison and reward computation
- `scripts/test_webcam.py` - CLI testing script (preview, baseline, full-test modes)
- `scripts/create_test_video.py` - Generate synthetic test video for WSL2 fallback

**Features**:
- Thread-safe webcam capture with automatic fallback to video file (WSL2 support)
- DeepFace emotion detection (7 emotions: angry, disgust, fear, happy, sad, surprise, neutral)
- Configurable emotion weights for amusement score calculation
- Temporal weighting with exponential decay for reaction scoring
- Clipped rewards in [-1, 1] range for RL stability

**Dependencies to add**:
```
opencv-python>=4.8.0
deepface>=0.0.79
tf-keras>=2.15.0
```

**Key classes**:

1. `WebcamCapture` - Thread-safe webcam with frame buffer
   - Background thread for continuous capture
   - Ring buffer to avoid frame buildup
   - WSL2 fallback to video file for testing
   - Context manager support (`with WebcamCapture() as cam`)

2. `EmotionAnalyzer` - DeepFace wrapper
   - Lazy loading of model (heavy dependency)
   - `analyze_frame()` / `analyze_frames()` for emotion detection
   - `capture_baseline()` - capture neutral state before meme
   - Configurable emotion weights for amusement score

3. `RewardCalculator` - Convert reactions to RL rewards
   - Formula: `reward = clip((reaction_score - baseline) * scale, -1, 1)`
   - Temporal weighting (recent frames weighted more)
   - Handles missing face detection gracefully

**Dataclasses**:
- `EmotionResult` - face_detected, emotions dict, dominant_emotion, amusement_score
- `BaselineResult` - avg_emotions, avg_amusement, frame_count
- `RewardResult` - reward, raw_delta, reaction_score, baseline_score

**WSL2 Considerations**:
- Option A: Use `usbipd wsl attach` to share USB camera
- Option B: Fallback to video file for testing (`data/test_video.mp4`)

### Phase 4: Meme Generation ✅ COMPLETE
**Goal**: Compose memes with text overlays and display with audio

**Files created**:
- `src/data/sound_manager.py` - Sound discovery and selection (like TemplateManager)
- `src/meme/__init__.py` - Module exports
- `src/meme/composer.py` - PIL-based text overlay with auto-sizing and wrapping
- `src/meme/player.py` - OpenCV display + pygame audio playback
- `scripts/test_meme.py` - CLI for compose/play/demo/list modes

**Features**:
- PIL-based text overlay with Impact-style stroke effect
- Automatic text wrapping and font sizing
- pygame audio playback (MP3/WAV/OGG)
- OpenCV window display with fade effects
- Synchronized audio/video playback
- Keyboard controls (ESC skip, Q quit)

**Dependencies added**: `pygame>=2.5.0`

**Key classes**:
1. `SoundManager` - Sound file discovery and random selection
2. `MemeComposer` - Text overlay on templates with styling
3. `MemePlayer` - Combined display + audio playback
4. `AudioPlayer` - Non-blocking pygame audio wrapper

**Usage**:
```bash
python scripts/test_meme.py --mode list      # List resources
python scripts/test_meme.py --mode compose   # Generate meme image
python scripts/test_meme.py --mode play      # Play single meme
python scripts/test_meme.py --mode demo -n 5 # Demo with 5 memes
```

### Phase 5: RL Pipeline ✅ COMPLETE
**Goal**: Train an RL agent to select optimal template+sound combinations

**Files created**:
- `src/rl/__init__.py` - Module exports
- `src/rl/environment.py` - Custom Gymnasium environment (MemeEnv)
- `src/rl/agent.py` - PPO wrapper with training callback
- `scripts/train_rl.py` - CLI for training, evaluation, and simulation

**Features**:
- MultiDiscrete action space: [template_idx, sound_idx] (captions excluded for now)
- Observation: one-hot encoded selections + 10-step reward history (279 dims)
- Exploration bonus (+0.1) for new template/sound combinations
- Repeat penalty (-0.05) to encourage variety
- PPO algorithm (handles noisy human rewards well)

**Dependencies added**:
```
gymnasium>=0.29.0
stable-baselines3>=2.2.0
torch>=2.10.0
```

**Key classes**:
1. `MemeEnv` - Gymnasium environment
   - `step(action)` - Returns observation, accepts (template_idx, sound_idx)
   - External reward via `set_external_reward()` for human feedback
   - Tracks combination exploration statistics
   - `get_stats()` returns coverage, avg reward, top combinations

2. `MemeAgent` - stable-baselines3 PPO wrapper
   - `train(timesteps, callback)` - Train the agent
   - `select_action(obs)` - Get next action
   - `save(path)` / `load(path)` - Model persistence

3. `TrainingCallback` - Progress tracking
   - Logs episode rewards and lengths
   - Progress bar with Rich console

**Usage**:
```bash
python scripts/train_rl.py --mode info           # Show action space info
python scripts/train_rl.py --mode simulate       # Train with random rewards
python scripts/train_rl.py --mode eval           # Evaluate saved model
python scripts/train_rl.py --timesteps 10000     # Custom training length
python scripts/train_rl.py --save data/models/my_model.zip
```

### Phase 6: Integration ✅ COMPLETE
**Goal**: Connect all components into a unified training pipeline

**Files created**:
- `src/pipeline/__init__.py` - Module exports
- `src/pipeline/orchestrator.py` - Main orchestrator class
- `scripts/train_interactive.py` - Interactive training CLI

**Features**:
- Unified training loop: baseline → show meme → capture reaction → compute reward → update agent
- Demo mode for testing without training
- Evaluation mode for assessing trained models
- Keyboard controls (ESC skip, Q quit)
- Automatic model saving during training
- Session statistics tracking

**Key classes**:
1. `MemeOrchestrator` - Main controller
   - `initialize()` - Sets up all components
   - `train(num_episodes)` - Full training session
   - `demo(num_memes)` - Random meme playback with reactions
   - `evaluate(num_memes)` - Assess model without updating

2. `OrchestratorConfig` - Configuration
   - Timing: baseline/meme duration, pause between memes
   - Display: window size, fullscreen
   - Webcam: camera device, fallback video
   - Training: episodes, save frequency

3. `SessionStats` - Statistics tracking
   - Total memes shown, episodes completed
   - Reward history and averages
   - Best combinations discovered

**Usage**:
```bash
# Demo mode (random memes, no training)
python scripts/train_interactive.py --mode demo --num-memes 5

# Training mode
python scripts/train_interactive.py --mode train --episodes 10 --model data/models/agent.zip

# Evaluate trained model
python scripts/train_interactive.py --mode eval --model data/models/agent.zip --num-memes 10

# With custom settings
python scripts/train_interactive.py --mode train --fullscreen --meme-duration 6
```

---

## 🎉 All Phases Complete!

The RL Meme Generator pipeline is now fully implemented:

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Sound Scraper | ✅ | 200+ sounds from myinstants.com |
| 2. Template Management | ✅ | 75 meme templates from Google Drive |
| 3. Facial Reaction | ✅ | DeepFace emotion detection + rewards |
| 4. Meme Generation | ✅ | Template display + audio playback |
| 5. RL Pipeline | ✅ | PPO agent with custom Gymnasium env |
| 6. Integration | ✅ | Orchestrator connecting all components |

---

## Immediate Implementation: Sound Scraper

### Step 1: Create project structure
Create directories and `requirements.txt`

### Step 2: Implement `sound_scraper.py`
```python
# Key class: MyInstantsScraper
# - Uses Selenium with Chrome in headless mode
# - Iterates through pages using ?page=N
# - Extracts sound names and detail URLs
# - Handles pagination until no more content
```

### Step 3: Implement `download_manager.py`
```python
# Key class: SoundDownloadManager
# - Resolves MP3 URLs from detail pages
# - Downloads with retry logic
# - Concurrent downloads with ThreadPoolExecutor
# - Sanitizes filenames
```

### Step 4: Create CLI script
```python
# scripts/scrape_sounds.py
# - Configurable max pages
# - Progress reporting
# - Resume capability
```

---

## Phase 2 Implementation: Template Loader

### Step 1: Update requirements.txt
Add: `gdown>=5.2.0`, `Pillow>=10.1.0`

### Step 2: Create `src/data/__init__.py`
Export DriveLoader, TemplateInfo, TemplateManager, Template

### Step 3: Implement `drive_loader.py`
```python
# Key class: DriveLoader
# - extract_folder_id() - parse URL to get folder ID
# - download_folder() - use gdown.download_folder()
# - save_metadata() - save templates.json
# - skip_existing for resume capability
```

### Step 4: Implement `template_manager.py`
```python
# Key class: TemplateManager
# - Lazy loading with _discover_templates()
# - get_random(n) - random selection for RL
# - filter_by_size(), filter_by_aspect_ratio()
# - Template dataclass with PIL Image loading
```

### Step 5: Create CLI script
```python
# scripts/download_templates.py
# - Default folder URL hardcoded
# - --url for custom folder
# - --output-dir, --no-skip, --quiet flags
```

### Verification
```bash
python scripts/download_templates.py
ls data/templates/  # Should show image files
```

---

## Phase 3 Implementation: Facial Reaction Scoring

### Step 1: Update requirements.txt
Uncomment/add:
```
opencv-python>=4.8.0
deepface>=0.0.79
tf-keras>=2.15.0
```

### Step 2: Create `src/vision/__init__.py`
Export all classes:
```python
from .webcam_capture import WebcamCapture, CaptureConfig
from .emotion_analyzer import EmotionAnalyzer, EmotionResult, BaselineResult, AnalyzerConfig
from .reward_calculator import RewardCalculator, RewardResult, RewardConfig

__all__ = [...]
```

### Step 3: Implement `webcam_capture.py`
```python
# Key class: WebcamCapture
# - Threading with deque buffer (maxlen for ring buffer)
# - _capture_loop() runs in background thread
# - get_frame() / get_frames() with thread lock
# - WSL2 fallback to video file if camera unavailable
# - Context manager for clean resource handling
```

### Step 4: Implement `emotion_analyzer.py`
```python
# Key class: EmotionAnalyzer
# - Lazy import of deepface (heavy)
# - analyze_frame() returns EmotionResult
# - capture_baseline(webcam, duration=3.0) captures neutral state
# - Configurable weights: happy=0.7, surprise=0.3
# - amusement_score property on EmotionResult
```

### Step 5: Implement `reward_calculator.py`
```python
# Key class: RewardCalculator
# - set_baseline(BaselineResult)
# - compute_reward(List[EmotionResult]) -> RewardResult
# - Temporal weighting with decay_factor=0.95
# - Configurable clipping: clip_min=-1, clip_max=1, scale=2.0
```

### Step 6: Create CLI test script
```bash
# scripts/test_webcam.py
python scripts/test_webcam.py --mode preview      # Show webcam with emotion overlay
python scripts/test_webcam.py --mode baseline     # Capture baseline
python scripts/test_webcam.py --mode full-test    # Complete pipeline test
python scripts/test_webcam.py --video-file x.mp4  # Use video instead of camera
```

### Verification
```bash
source .venv/bin/activate
pip install opencv-python deepface tf-keras
python scripts/test_webcam.py --mode preview
# Should show webcam feed with emotion detection overlay
# Press 'q' to quit
```

---

## User Preferences
- **Sound limit**: 50-100 sounds (quick test set)
- **Captions**: None initially (just images + sounds)
- **Browser mode**: Headless (background execution)

---

## Tech Stack
| Component | Technology |
|-----------|------------|
| Web Scraping | requests, BeautifulSoup |
| Google Drive | gdown (simpler than PyDrive2) |
| Image Processing | Pillow, OpenCV |
| Audio | pygame, moviepy |
| Emotion Detection | DeepFace (with MTCNN) |
| RL Framework | Stable-Baselines3, Gymnasium |
| Deep Learning | PyTorch |

---

## Verification Plan
1. **Sound Scraper**: Run scraper, verify MP3 files are downloaded and playable
2. **Emotion Detection**: Test webcam capture and emotion scores in real-time
3. **End-to-End**: Run training session, verify rewards correlate with facial expressions