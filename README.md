# SignBridge Words - Indian Sign Language words & phrases -> English (live, with AR bubble)

# QUICK START - NO RECORDING (public INCLUDE dataset only)

Files you use: prepare_include.py -> extract_videos.py -> train_words.py -> translate.py
(core.py and ar_ui.py are used by them automatically. record_words.py is optional - skip it.)

1. Download a few parts from https://zenodo.org/records/4010759  (CC-BY 4.0, credit the authors).
   The full set is 56.8 GB - do NOT download it all. Good small start (~5 GB):
   Greetings_1of2.zip, Greetings_2of2.zip, Pronouns_1of2.zip, Pronouns_2of2.zip
   Add more later (People_*, Society_*, Adjectives_*, ...) for a bigger vocabulary.
   Put the .zip files in a folder called  downloads/
2. Unzip all parts into one folder and see which words you have:
       python prepare_include.py --zips downloads --out INCLUDE
3. Convert videos to landmarks (slowest step; start small):
       python extract_videos.py --data INCLUDE --out clips_include --top-n 40
   (--top-n 40 = the 40 words with the most videos. Or pick words from include_words.csv:
    --classes "hello,thank you,..." )
4. Train:
       python train_words.py --data clips_include
5. Run:
       python translate.py


Raise your hands and sign. When you pause, the app recognises the word/phrase, shows it in a speech
bubble beside your hand (follows it, smoothed), adds it to a sentence bar and speaks it.

## 1. Setup (VS Code terminal) - Python 3.9 to 3.12
    python -m venv .venv
    .venv\Scripts\activate            (Mac/Linux: source .venv/bin/activate)
    pip install -r requirements.txt

## 2. Test tracking + the AR bubble first (no data needed)
    python translate.py
Hand up -> "signing..." -> lower hands -> bubble shows "SIGN". Press h to toggle the skeleton.
Android phone as camera: install "IP Webcam", Start server, then
    python translate.py --source http://PHONE_IP:8080/video --mirror off

## 3. Get training data (pick A, B or both)

### A) Record your own words and PHRASES (recommended for phrases)
Every word or phrase is one class.
    python record_words.py --words "hello,thank you,how are you,i need help,water"
SPACE = start (2 s countdown) / stop. Record 25+ clips per phrase: vary speed, distance, lighting,
clothes - and have other people sign too. Saved to clips_own/.

### B) Use a public ISL video dataset (words)
- INCLUDE (IIT Madras / AI4Bharat): ~4,292 videos, 263 word signs, 15 categories
  (Zenodo record 4010759, and sign-language.ai4bharat.org). The full set is very large - download only the
  parts/words you need.
- CISLR (EMNLP 2022): word-level ISL videos, ~4,700 words.
- ISL-CSLTR (Mendeley Data, kcmpdxky7p): sentence-level, 100 sentences, 7 signers - check what
  format it ships in; sentence = one class works the same way.
Always check each dataset's licence. Put videos in folders named by the word (e.g. INCLUDE/.../loud/*.MOV), then
    python extract_videos.py --data INCLUDE --out clips_include --classes "hello,thank you,help,water,food"
(drop --classes to use everything; add --max-per-class 30 --workers 4 to go faster)

## 4. Train
    python train_words.py --data clips_own clips_include
Prints top-1 and top-3 accuracy on held-out clips (never augmented, never seen in training) and the
most confused word pairs. Saves models/words.joblib. Add --models svm mlp rf to compare all three.

## 5. Run
    python translate.py
Keys: SPACE manual start/stop (if auto-detect misbehaves) | c clear | s speak sentence |
BACKSPACE undo | h skeleton | v voice | q quit.  Options: --pause 0.5  --min-conf 0.55  --no-voice

## Tips for accuracy
- Camera at chest/face height, upper body + both hands visible, plain-ish background, good light.
- Train on the same camera setup you will demo with, and include your own recordings even if you use INCLUDE.
- Adding the clips of the SAME people you will demo with helps the most.
- Similar-looking signs get confused: check the "most confused" list and record more clips of those.

## Honest limits
- It recognises only the words/phrases you trained, one at a time (isolated signs), not free continuous signing.
- Real ISL has regional variation; results on a new signer depend on how varied the training data is.
- This classical-ML model (PCA+SVM / MLP on landmark sequences) is light and trains in minutes on a laptop.
  Published deep-learning models on all 263 INCLUDE words report ~85%, so expect lower on a big vocabulary
  and much better on a focused 20-50 word set.
