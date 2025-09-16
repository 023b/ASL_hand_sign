#app.py

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import difflib
from string import ascii_uppercase

# Try to use the 'enchant' dictionary library for word suggestions if available

try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False

# A fallback list of common words in case enchant is not available

FALLBACK_WORDS = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he',
    'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
    'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
    'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
    'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
    'day', 'most', 'us', 'hello', 'world', 'thanks', 'bye'
]


class Application:
    def __init__(self, model_dir='model'):
         # Store the directory where trained models are located
        self.directory = model_dir

         # Start capturing video feed from the default camera (index 0)
        self.vs = cv2.VideoCapture(0)
        # Load machine learning models for hand sign recognition
        self._load_models()
        # Setup backend for word suggestions (spell checker or fallback)
        self._init_suggestion_backend()

        # Initialize runtime state variables
        self.ct = {char: 0 for char in list(ascii_uppercase) + ['blank']}
        self.sentence = ""
        self.word = ""
        self.current_symbol = "..."
        self.confidence = 0.0
        self.history = []
        self.char_accepted_flag = False

        # --- UI Setup ---
        # FIX: The root window MUST be created before any fonts are defined.
        self._setup_tk_root()
        self._setup_theme_and_fonts()
        self._setup_ui_layout()
        self._bind_keyboard_shortcuts()
        
        # Load the signs image after the main window is drawn and has a size
        self.root.after(200, self._load_and_display_signs_image)

        # --- Start the main loop ---
        self.video_loop()

    def _load_models(self):
        """Loads all Keras models."""
        def load_single_model(json_path, weights_path):
            if not os.path.exists(json_path) or not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing model files: {json_path} or {weights_path}")
            with open(json_path, 'r') as jf:
                model_json = jf.read()
            model = model_from_json(model_json)
            model.load_weights(weights_path)
            return model

        try:
            self.loaded_model = load_single_model(os.path.join(self.directory, 'model-bw.json'), os.path.join(self.directory, 'model-bw.h5'))
            self.loaded_model_dru = load_single_model(os.path.join(self.directory, 'model-bw_dru.json'), os.path.join(self.directory, 'model-bw_dru.h5'))
            print("All models loaded successfully.")
        except Exception as e:
            print(f"FATAL: Model loading failed. Error: {e}")
            raise

    def _init_suggestion_backend(self):
        """Initializes the word suggestion engine."""
        self.suggestion_engine = None
        if ENCHANT_AVAILABLE:
            try:
                self.suggestion_engine = enchant.Dict('en_US')
                print("Using 'enchant' for word suggestions.")
            except Exception as e:
                print(f"Could not start enchant: {e}")
        if self.suggestion_engine is None:
            print("Using fallback word list for suggestions.")

    def _setup_tk_root(self):
        """Initializes the main Tkinter window."""
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Translator")
        self.root.configure(bg='#000000')
        self.root.state('zoomed')
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

    def _setup_theme_and_fonts(self):
        """Defines the color scheme and fonts for the UI."""
        self.BG_COLOR = '#000000'
        self.TEXT_COLOR = '#FFFFFF'
        self.ACCENT_COLOR = '#00E676'
        self.PANEL_BG_COLOR = '#121212'
        self.BUTTON_COLOR = '#222222'

        self.large_font = tkfont.Font(family="Segoe UI", size=24, weight="bold")
        self.medium_font = tkfont.Font(family="Segoe UI", size=16)
        self.small_font = tkfont.Font(family="Segoe UI", size=12)
        self.hud_font = tkfont.Font(family="Consolas", size=11)

    def _setup_ui_layout(self):
        """Creates and places all the UI widgets."""
        # Main content frame
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side (video and text)
        left_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.panel = tk.Label(left_frame, bg=self.PANEL_BG_COLOR) # Video Feed
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        text_frame = tk.Frame(left_frame, bg=self.BG_COLOR, pady=15)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X)
        text_frame.grid_columnconfigure(1, weight=1)

        tk.Label(text_frame, text="Detected:", font=self.medium_font, bg=self.BG_COLOR, fg=self.ACCENT_COLOR).grid(row=0, column=0, sticky='nw', pady=2)
        self.current_symbol_label = tk.Label(text_frame, text=self.current_symbol, font=self.large_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.current_symbol_label.grid(row=0, column=1, sticky='nw', padx=10)

        tk.Label(text_frame, text="Word:", font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR).grid(row=1, column=0, sticky='nw', pady=2)
        self.word_label = tk.Label(text_frame, text=self.word, font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR)
        self.word_label.grid(row=1, column=1, sticky='nw', padx=10)

        tk.Label(text_frame, text="Sentence:", font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR).grid(row=2, column=0, sticky='nw', pady=2)
        self.sentence_label = tk.Label(text_frame, text=self.sentence, font=self.medium_font, bg=self.BG_COLOR, fg=self.TEXT_COLOR, wraplength=int(self.root.winfo_screenwidth() * 0.5), justify='left')
        self.sentence_label.grid(row=2, column=1, sticky='nw', padx=10)
        
        self.hud_label = tk.Label(text_frame, text="", font=self.hud_font, bg=self.BG_COLOR, fg='#888888', justify='left')
        self.hud_label.grid(row=3, column=0, columnspan=2, sticky='nw', pady=(20, 0))
        
        # Right side (processed feed, suggestions, signs image)
        right_frame = tk.Frame(main_frame, bg=self.BG_COLOR, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_frame.pack_propagate(False)

        self.panel2 = tk.Label(right_frame, bg=self.PANEL_BG_COLOR) # Processed Feed
        self.panel2.pack(side=tk.TOP, fill=tk.X)
        
        sugg_frame = tk.Frame(right_frame, bg=self.BG_COLOR)
        sugg_frame.pack(side=tk.TOP, fill=tk.X, pady=20)
        tk.Label(sugg_frame, text="Suggestions", font=self.medium_font, bg=self.BG_COLOR, fg=self.ACCENT_COLOR).pack()
        
        self.suggestion_buttons = []
        for i in range(5):
            btn = tk.Button(sugg_frame, text="", font=self.small_font, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR,
                            relief='flat', command=lambda idx=i: self._use_suggestion(idx))
            btn.pack(fill='x', padx=5, pady=3)
            self.suggestion_buttons.append(btn)
        
        self.signs_panel = tk.Label(right_frame, bg=self.BG_COLOR) # Signs Reference
        self.signs_panel.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _bind_keyboard_shortcuts(self):
        self.root.bind('<space>', self._commit_word)
        self.root.bind('<BackSpace>', self._delete_char)
        self.root.bind('c', self._clear_sentence)
        self.root.bind('<Escape>', lambda e: self.destructor())

    def _load_and_display_signs_image(self):
        candidate_path = 'signs.png'
        if not os.path.exists(candidate_path):
             candidate_path = os.path.join('Sign-Language-to-Text-master', 'signs.png')
             
        if os.path.exists(candidate_path):
            try:
                # Ensure the panel is drawn and has a size
                self.signs_panel.update_idletasks() 
                pw = self.signs_panel.winfo_width()
                ph = self.signs_panel.winfo_height()

                if pw > 1 and ph > 1:
                    img = Image.open(candidate_path).convert('RGBA')
                    img.thumbnail((pw, ph), Image.LANCZOS)
                    canvas = Image.new('RGBA', (pw, ph), (0, 0, 0, 255))
                    x = (pw - img.width) // 2
                    y = (ph - img.height) // 2
                    canvas.paste(img, (x, y), img)
                    self.signs_image_tk = ImageTk.PhotoImage(canvas)
                    self.signs_panel.config(image=self.signs_image_tk)
            except Exception as e:
                print(f"Error loading signs.png: {e}")
        else:
            print("Warning: 'signs.png' not found.")
    
    def _commit_word(self, event=None):
        if self.word:
            self.sentence += (" " if self.sentence else "") + self.word
            self.word = ""
            self._update_text_labels()
            
    def _delete_char(self, event=None):
        if self.word:
            self.word = self.word[:-1]
            self._update_text_labels()

    def _clear_sentence(self, event=None):
        self.sentence = ""
        self.word = ""
        self._update_text_labels()
        
    def _use_suggestion(self, idx):
        text = self.suggestion_buttons[idx].cget('text')
        if text:
            self.word = text.upper()
            self._commit_word()

    def video_loop(self):
        ok, frame = self.vs.read()
        if not ok:
            print("Camera feed lost.")
            self.destructor()
            return

        frame = cv2.flip(frame, 1)
        
        height, width, _ = frame.shape
        roi_size = int(min(height, width) * 0.4)
        x1 = width - roi_size - 20
        y1 = 20
        x2 = width - 20
        y2 = y1 + roi_size

        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 230, 118), 2)
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, processed_roi = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self.predict(processed_roi)

        img2 = Image.fromarray(processed_roi)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        self.panel2.imgtk = imgtk2
        self.panel2.config(image=imgtk2)
        
        self._update_text_labels()
        self._update_suggestions()
        self._update_hud()

        self.root.after(20, self.video_loop)

    def predict(self, test_image):
        if test_image is None or test_image.size == 0:
            return

        test_image = cv2.resize(test_image, (128, 128))
        arr = test_image.reshape(1, 128, 128, 1).astype('float32') / 255.0

        result = self.loaded_model.predict(arr, verbose=0)
        prediction = {ascii_uppercase[i]: float(result[0][i+1]) for i in range(26)}
        prediction['blank'] = float(result[0][0])
        
        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_symbol = prediction_sorted[0][0]
        top_prob = prediction_sorted[0][1]

        if top_symbol in ('D', 'R', 'U'):
            result_dru = self.loaded_model_dru.predict(arr, verbose=0)
            pred2 = {'D': float(result_dru[0][0]), 'R': float(result_dru[0][1]), 'U': float(result_dru[0][2])}
            top_symbol = max(pred2, key=pred2.get)
            top_prob = pred2[top_symbol]

        self.current_symbol = top_symbol
        self.confidence = float(top_prob)

        # --- Improved Character Acceptance Logic ---
        THRESHOLD_COUNT = 15
        
        if top_symbol != 'blank':
            self.ct[top_symbol] += 1
            if self.ct[top_symbol] > THRESHOLD_COUNT and not self.char_accepted_flag:
                if len(self.history) == 0 or self.history[-1] != top_symbol:
                    self.word += top_symbol
                    self.history.append(top_symbol)
                self.char_accepted_flag = True # Mark as accepted for this gesture
        else:
            # Reset all counters and the acceptance flag when a 'blank' is seen
            for char in self.ct: self.ct[char] = 0
            self.char_accepted_flag = False

    def _update_text_labels(self):
        self.current_symbol_label.config(text=self.current_symbol)
        self.word_label.config(text=self.word)
        self.sentence_label.config(text=self.sentence)

    def _update_suggestions(self):
        suggestions = []
        if self.word:
            prefix = self.word.lower()
            if self.suggestion_engine:
                suggestions = self.suggestion_engine.suggest(prefix)[:5]
            else:
                prefix_matches = [w for w in FALLBACK_WORDS if w.startswith(prefix)]
                suggestions = prefix_matches[:5]
                if len(suggestions) < 5:
                    close_matches = difflib.get_close_matches(prefix, FALLBACK_WORDS, n=5-len(suggestions), cutoff=0.7)
                    suggestions.extend([m for m in close_matches if m not in suggestions])
        
        for i, btn in enumerate(self.suggestion_buttons):
            btn.config(text=suggestions[i].capitalize() if i < len(suggestions) else "")

    def _update_hud(self):
        hist_str = "".join(self.history[-10:])
        hud_text = (
            f"Confidence: {self.confidence:.2f}   |   History: {hist_str}\n"
            f"[Space] Add Word | [Backspace] Del Char | [c] Clear All"
        )
        self.hud_label.config(text=hud_text)

    def destructor(self):
        print("Closing application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"An error occurred during application startup: {e}")