# Sign Language Recognition - Project Workflow & Architecture

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Collection│    │   Preprocessing  │    │  Model Training │
│   (collect-data.py) │ → │ (preprocessing.py)│ → │   (train.py)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Real-time Application                        │
│                        (app.py)                                │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Workflow Process

### Phase 1: Data Collection and Preparation

#### Step 1: Raw Data Collection (`collect-data.py`)
```
User Input → Camera Capture → ROI Extraction → Image Processing → File Storage

Flow Details:
1. Initialize camera feed
2. Display live video with ROI overlay
3. User performs sign language gestures
4. Press corresponding key (A-Z, 0-2) to capture
5. Apply preprocessing pipeline:
   - Convert to grayscale
   - Gaussian blur (5x5)
   - Adaptive thresholding
   - OTSU binary thresholding
6. Save processed image to appropriate folder
7. Update real-time counters on screen
```

**Directory Structure Created:**
```
data/
├── train/
│   ├── A/ (images for letter A)
│   ├── B/ (images for letter B)
│   ├── ...
│   ├── Z/ (images for letter Z)
│   ├── 0/ (images for number 0)
│   ├── 1/ (images for number 1)
│   └── 2/ (images for number 2)
└── test/
    ├── A/
    ├── B/
    ├── ...
    └── 2/
```

#### Step 2: Dataset Preprocessing (`preprocessing.py`)
```
Raw Images → Batch Processing → Train/Test Split → Processed Dataset

Flow Details:
1. Scan all directories in raw data folder
2. For each image file:
   - Load original image
   - Apply standardized processing (func from image_processing.py)
   - Determine train/test allocation
   - Save to processed data directory (data2/)
3. Generate statistics (total processed, train count, test count)
```

**Processing Pipeline:**
```
Original Image (Color) 
    ↓
Grayscale Conversion
    ↓
Gaussian Blur (5x5, σ=2)
    ↓
Adaptive Thresholding (GAUSSIAN_C, 11x11)
    ↓
OTSU Binary Thresholding (threshold=70)
    ↓
Processed Binary Image (128x128)
```

### Phase 2: Model Development and Training

#### Step 3: CNN Model Training (`train.py`)
```
Processed Dataset → Data Augmentation → Model Training → Model Saving

Architecture Flow:
Input (128x128x1) 
    ↓
Conv2D(32, 3x3, ReLU) → MaxPool2D(2x2)
    ↓
Conv2D(32, 3x3, ReLU) → MaxPool2D(2x2)
    ↓
Flatten → Dense(128, ReLU) → Dropout(0.4)
    ↓
Dense(96, ReLU) → Dropout(0.4)
    ↓
Dense(64, ReLU) → Dense(27, Softmax)
    ↓
Output (27 classes: A-Z + blank)
```

**Training Configuration:**
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 10 images per batch
- **Epochs**: 5 complete dataset passes
- **Data Augmentation**: Shear (0.2), Zoom (0.2), Horizontal Flip

**Output Files:**
- `model-bw.json`: Model architecture
- `model-bw.h5`: Trained weights
- `model-bw_dru.json`: Specialized D/R/U model architecture  
- `model-bw_dru.h5`: Specialized D/R/U model weights

### Phase 3: Real-Time Recognition System

#### Step 4: Application Runtime (`app.py`)
```
Camera Input → Image Processing → Model Prediction → Text Output → GUI Update

Detailed Runtime Flow:

1. INITIALIZATION:
   ┌─ Load trained models (primary + D/R/U specialist)
   ├─ Initialize camera capture
   ├─ Setup GUI components
   ├─ Initialize word suggestion engine
   └─ Start video processing loop

2. REAL-TIME PROCESSING LOOP (50 FPS):
   ┌─ Capture frame from camera
   ├─ Mirror image horizontally
   ├─ Define ROI (Region of Interest)
   ├─ Extract ROI from frame
   ├─ Apply preprocessing pipeline
   ├─ Run model prediction
   ├─ Apply character acceptance logic
   ├─ Update text displays
   ├─ Generate word suggestions
   ├─ Update GUI elements
   └─ Schedule next iteration (20ms delay)

3. PREDICTION PIPELINE:
   Input Image (ROI) → Resize(128x128) → Normalize(0-1) 
   → Primary Model → Confidence Scores → Character Selection
   → [If D/R/U detected] → Secondary Model → Final Character
   → Character Acceptance Logic → Text Update

4. CHARACTER ACCEPTANCE LOGIC:
   ┌─ Increment counter for detected character
   ├─ Check if counter > threshold (15)
   ├─ Verify character not recently added (duplicate prevention)
   ├─ Add character to current word
   ├─ Update display
   └─ [If blank detected] → Reset all counters
```

## System Integration Architecture

### Data Flow Diagram
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Camera    │────▶│ Frame Capture│────▶│ ROI Extract │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                    ┌─────────────┐     ┌──────▼──────┐
                    │ GUI Display │◀────│Preprocessing│
                    └─────────────┘     └─────────────┘
                            ▲                   │
                    ┌───────┴────────┐  ┌───────▼───────┐
                    │ Text Processing│◀─│Model Prediction│
                    └────────────────┘  └──────────────┘
                            ▲                   │
                    ┌───────┴────────┐  ┌───────▼──────┐
                    │Word Suggestions│  │  Confidence  │
                    └────────────────┘  │  Evaluation  │
                                        └──────────────┘
```

### Component Interaction Model

#### 1. **Model Management System**
- **Primary Model**: Handles 26 letters + blank recognition
- **Secondary Model**: Disambiguates similar signs (D, R, U)
- **Model Loading**: JSON architecture + H5 weights
- **Prediction Pipeline**: Preprocessing → Inference → Post-processing

#### 2. **Image Processing Pipeline**
- **Capture Module**: OpenCV camera interface
- **ROI Detection**: Dynamic region of interest
- **Preprocessing**: Grayscale → Blur → Thresholding → Normalization
- **Format Conversion**: BGR → RGB → PIL → Tkinter PhotoImage

#### 3. **User Interface System**
- **Video Display**: Real-time camera feed with ROI overlay
- **Processed View**: Binary image showing model input
- **Text Output**: Current character, word, sentence displays
- **Suggestions Panel**: Real-time word completion suggestions
- **Reference Display**: ASL alphabet reference image

#### 4. **Text Processing Engine**
- **Character Assembly**: Individual letters → words → sentences
- **Duplicate Prevention**: Gesture history tracking
- **Word Suggestions**: PyEnchant dictionary + fallback word list
- **User Interactions**: Keyboard shortcuts for text manipulation

#### 5. **State Management**
```python
Application State Variables:
├── current_symbol: Currently detected character
├── confidence: Prediction confidence score
├── word: Currently building word
├── sentence: Complete sentence text  
├── history: Recent character detection history
├── ct: Character detection counters
└── char_accepted_flag: Prevents duplicate character addition
```

## Performance Optimization Strategies

### 1. **Real-Time Processing Optimizations**
- **Frame Rate Control**: 50 FPS processing with 20ms intervals
- **Efficient Memory Usage**: In-place image operations
- **Model Caching**: Single model loading at startup
- **ROI Optimization**: Process only relevant image regions

### 2. **Accuracy Enhancement Techniques**
- **Threshold Counting**: Require multiple consistent detections
- **Dual Model System**: Specialized disambiguation for similar signs
- **Confidence Scoring**: Reject low-confidence predictions
- **Gesture History**: Prevent duplicate character insertion

### 3. **User Experience Features**
- **Visual Feedback**: Real-time confidence and detection display
- **Word Suggestions**: Intelligent word completion
- **Keyboard Shortcuts**: Efficient text manipulation
- **Reference Display**: Always-visible ASL alphabet guide

## Error Handling and Robustness

### 1. **Hardware Failure Management**
- Camera disconnection detection
- Graceful degradation for missing models
- Resource cleanup on application exit

### 2. **Model Robustness**
- Fallback word suggestion system
- Exception handling in prediction pipeline
- Input validation and sanitization

### 3. **User Input Validation**
- Keyboard shortcut error handling
- GUI event management
- File path validation for model loading

This workflow ensures a complete, robust, and user-friendly sign language recognition system with high accuracy and real-time performance.
