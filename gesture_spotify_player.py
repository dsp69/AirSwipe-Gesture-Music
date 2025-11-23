#!/usr/bin/env python3
"""
Gesture-controlled music player (Spotify first, fallback to local).
Controls:
 - Fist: toggle play/pause
 - Swipe right: next track
 - Swipe left: previous track
 - Two fingers up (index+middle): volume (vertical position -> 0..100)
"""

import os
import time
import argparse
from collections import deque
import cv2
import numpy as np
import math
import pygame

try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe is required: pip install mediapipe")

# Optional Spotify
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIFY = True
except Exception:
    spotipy = None
    HAS_SPOTIFY = False

# ---------------- Hand Detector ----------------
class HandDetector:
    def __init__(self, max_num_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = [(pt.x, pt.y, pt.z) for pt in hand_landmarks.landmark]
                cx = int(np.mean([p[0] for p in lm]) * w)
                cy = int(np.mean([p[1] for p in lm]) * h)
                hands_data.append({'lm': lm, 'center': (cx, cy), 'raw': hand_landmarks})
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame, hands_data

# ---------------- Gesture Recognizer ----------------
class GestureRecognizer:
    def __init__(self, buffer_len=6, swipe_vpx=450.0, cooldown=0.9):
        self.center_buf = deque(maxlen=buffer_len)
        self.finger_buf = deque(maxlen=buffer_len)
        self.last_trigger = {}
        self.swipe_vpx = swipe_vpx
        self.cooldown = cooldown

    def fingers_up(self, lm):
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            try:
                if lm[tip][1] < lm[tip - 2][1]:
                    count += 1
            except:
                pass
        try:
            if abs(lm[4][0] - lm[0][0]) > 0.06:
                count += 1
        except:
            pass
        return count

    def smooth_count(self, cnt):
        self.finger_buf.append(cnt)
        return int(round(np.median(list(self.finger_buf))))

    def add_center(self, center):
        self.center_buf.append((center[0], center[1], time.time()))

    def detect_swipe(self):
        if len(self.center_buf) < 3:
            return None
        x0, y0, t0 = self.center_buf[0]
        x1, y1, t1 = self.center_buf[-1]
        dt = max(1e-3, t1 - t0)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        if abs(vx) > self.swipe_vpx and abs(vy) < abs(vx) * 0.5:
            return 'right' if vx > 0 else 'left'
        return None

    def cooldown_ok(self, action):
        now = time.time()
        last = self.last_trigger.get(action, 0)
        if now - last >= self.cooldown:
            self.last_trigger[action] = now
            return True
        return False

    def recognize(self, hand):
        if not hand:
            return None, None
        lm = hand['lm']
        center = hand['center']
        self.add_center(center)

        raw = self.fingers_up(lm)
        cnt = self.smooth_count(raw)

        # volume gesture: index + middle up, ring & pinky down
        if cnt >= 2:
            try:
                iu = lm[8][1] < lm[6][1]
                mu = lm[12][1] < lm[10][1]
                ru = lm[16][1] < lm[14][1]
                pu = lm[20][1] < lm[18][1]
                if iu and mu and not ru and not pu:
                    vol_norm = 1.0 - np.mean([lm[8][1], lm[12][1]])
                    vol = int(np.clip(vol_norm * 100, 0, 100))
                    return 'volume', vol
            except:
                pass

        # fist detection
        folded = 0
        for tip in [4, 8, 12, 16, 20]:
            try:
                if lm[tip][1] > lm[tip - 2][1]:
                    folded += 1
            except:
                pass
        if folded >= 4:
            return 'fist', None

        # swipe detection
        sw = self.detect_swipe()
        if sw == 'right':
            return 'swipe_right', None
        if sw == 'left':
            return 'swipe_left', None

        return None, None

# ---------------- Spotify Controller ----------------
class SpotifyController:
    def __init__(self):
        if spotipy is None:
            raise RuntimeError("Spotipy not installed.")
        # auth uses env vars or default browser flow
        self.auth_manager = SpotifyOAuth(open_browser=True,
                                         scope="user-modify-playback-state user-read-playback-state user-read-currently-playing")
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

    def is_available(self):
        try:
            return self.sp.current_playback() is not None
        except:
            return False

    def toggle_play_pause(self):
        try:
            cur = self.sp.current_playback()
            if cur and cur.get('is_playing'):
                self.sp.pause_playback()
            else:
                self.sp.start_playback()
            return True
        except Exception as e:
            print("Spotify toggle error:", e)
            return False

    def next(self):
        try:
            self.sp.next_track()
            return True
        except Exception as e:
            print("Spotify next error:", e)
            return False

    def previous(self):
        try:
            self.sp.previous_track()
            return True
        except Exception as e:
            print("Spotify prev error:", e)
            return False

    def set_volume(self, v):
        try:
            self.sp.volume(int(v))
            return True
        except Exception as e:
            print("Spotify set volume error:", e)
            return False

    def currently_playing(self):
        try:
            cur = self.sp.current_playback()
            if cur and cur.get('item'):
                name = cur['item']['name']
                artists = ', '.join([a['name'] for a in cur['item']['artists']])
                return f"{name} - {artists}"
            return None
        except:
            return None

# ---------------- Local Controller ----------------
class LocalController:
    def __init__(self, folder='local_music'):
        pygame.mixer.init()
        self.folder = folder
        self.tracks = []
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.mp3', '.wav', '.ogg')):
                    self.tracks.append(os.path.join(folder, f))
        self.idx = 0
        self.paused = True
        self.volume = 0.5
        pygame.mixer.music.set_volume(self.volume)
        if self.tracks:
            try:
                pygame.mixer.music.load(self.tracks[self.idx])
            except Exception as e:
                print("pygame load error:", e)

    def is_available(self):
        return len(self.tracks) > 0

    def toggle_play_pause(self):
        if not self.tracks:
            return False
        if pygame.mixer.music.get_busy() and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
        else:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
            else:
                pygame.mixer.music.play()
                self.paused = False
        return True

    def next(self):
        if not self.tracks:
            return False
        self.idx = (self.idx + 1) % len(self.tracks)
        pygame.mixer.music.load(self.tracks[self.idx])
        pygame.mixer.music.play()
        self.paused = False
        return True

    def previous(self):
        if not self.tracks:
            return False
        self.idx = (self.idx - 1) % len(self.tracks)
        pygame.mixer.music.load(self.tracks[self.idx])
        pygame.mixer.music.play()
        self.paused = False
        return True

    def set_volume(self, v):
        # v range 0-100
        vol = max(0, min(100, int(v))) / 100.0
        self.volume = vol
        pygame.mixer.music.set_volume(self.volume)
        return True

    def currently_playing(self):
        if not self.tracks:
            return None
        return os.path.basename(self.tracks[self.idx])

# ---------------- Overlay Drawing ----------------
def draw_overlay(frame, text, vol=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, text or '', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if vol is not None:
        cv2.rectangle(frame, (w - 140, 10), (w - 20, 30), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 140, 10), (w - 140 + int(vol / 100 * 120), 30), (50, 220, 50), -1)
        cv2.putText(frame, f'{vol} %', (w - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_instrument_panel(frame, instruments, current_idx, vols=None, play_pulse=False, t=0.0):
    h, w = frame.shape[:2]
    pad = 12
    panel_h = 90
    y0 = h - panel_h - pad
    cv2.rectangle(frame, (0, y0), (w, h), (10, 10, 10), -1)
    n = len(instruments)
    if n == 0:
        return
    slot_w = min(180, int((w - 2 * pad) / n))
    start_x = (w - (slot_w * n)) // 2
    for i, name in enumerate(instruments):
        x = start_x + i * slot_w
        y = y0 + 10
        color = (80, 80, 80)
        if i == current_idx:
            color = (70, 180, 70)
        cv2.rectangle(frame, (x + 6, y), (x + slot_w - 6, y + 56), color, -1)
        cv2.putText(frame, name, (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if vols is not None and i < len(vols):
            vv = int(vols[i])
            bx0 = x + 12
            by0 = y + 36
            bx1 = x + slot_w - 18
            by1 = y + 46
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), (50, 50, 50), -1)
            fill_w = int((bx1 - bx0) * (vv / 100.0))
            cv2.rectangle(frame, (bx0, by0), (bx0 + fill_w, by1), (50, 220, 50), -1)
    if play_pulse:
        cx = start_x + current_idx * slot_w + slot_w // 2
        cy = y0 + 28
        pulse = int(8 + 6 * (0.5 + 0.5 * math.sin(t * 6.0)))
        cv2.circle(frame, (cx, cy), 28 + pulse, (0, 200, 0), 2)

# ---------------- Main app ----------------
def main(debug=False):
    print('Starting gesture-controlled local player (debug=' + str(debug) + ')')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    detector = HandDetector()
    recognizer = GestureRecognizer()

    # Initialize controllers: try Spotify first (if installed & env present), fallback to local
    controller = None
    using = None

    if not debug:
        # Try Spotify if spotipy is installed and env vars exist
        if HAS_SPOTIFY and os.environ.get('SPOTIPY_CLIENT_ID') and os.environ.get('SPOTIPY_CLIENT_SECRET') and os.environ.get('SPOTIPY_REDIRECT_URI'):
            try:
                spc = SpotifyController()
                if spc.is_available():
                    controller = spc
                    using = 'spotify'
                    print("Initialized Spotify controller (using Spotify).")
                else:
                    # Spotify SDK working but no active device; still allow fallback to local
                    controller = spc
                    using = 'spotify'
                    print("Spotify controller initialized (no active device detected).")
            except Exception as e:
                print("Spotify init error, falling back to local:", e)

        # if Spotify not available, use local controller
        if controller is None:
            try:
                lc = LocalController()
                controller = lc
                using = 'local'
                if lc.is_available():
                    print("Using local playback from", lc.folder)
                else:
                    print("Local controller active but no tracks found in local_music/ (place files to enable).")
            except Exception as e:
                print("Local controller init error:", e)
                controller = None
    else:
        print("Debug mode: gestures printed, no audio controller initialized.")

    gesture_map = {
        'fist': 'toggle',
        'swipe_right': 'next',
        'swipe_left': 'prev',
        'volume': 'volume'
    }

    gesture_text = ''
    current_volume = None
    cooldown_indicator = ''
    instruments = ['Drums', 'Bass', 'Guitar', 'Piano', 'Synth']
    current_instrument = 0
    instrument_vols = [60 for _ in instruments]
    last_action_time = 0
    action_text = ''
    play_pulse = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            out_frame, hands = detector.find_hands(frame, draw=True)
            hand = hands[0] if hands else None

            gesture, data = recognizer.recognize(hand)

            overlay_text = ''

            if gesture == 'volume':
                vol = data
                if controller:
                    try:
                        controller.set_volume(vol)
                    except Exception:
                        pass
                overlay_text = f'VOLUME {vol}%'
                current_volume = vol
                instrument_vols[current_instrument] = vol
                print(f'[GESTURE] volume -> {vol}%')

            elif gesture == 'fist' and recognizer.cooldown_ok('toggle'):
                ok = False
                if controller:
                    try:
                        ok = controller.toggle_play_pause()
                    except Exception:
                        ok = False
                overlay_text = 'PAUSE/PLAY' if ok else 'TOGGLE FAILED'
                action_text = 'PAUSE/PLAY'
                last_action_time = time.time()
                play_pulse = True
                print('[GESTURE] fist -> toggle play/pause')

            elif gesture == 'swipe_right' and recognizer.cooldown_ok('next'):
                ok = False
                if controller:
                    try:
                        ok = controller.next()
                    except Exception:
                        ok = False
                overlay_text = 'NEXT' if ok else 'NEXT FAILED'
                current_instrument = (current_instrument + 1) % len(instruments)
                action_text = f'NEXT -> {instruments[current_instrument]}'
                last_action_time = time.time()
                play_pulse = False
                print('[GESTURE] swipe right -> next track')

            elif gesture == 'swipe_left' and recognizer.cooldown_ok('prev'):
                ok = False
                if controller:
                    try:
                        ok = controller.previous()
                    except Exception:
                        ok = False
                overlay_text = 'PREVIOUS' if ok else 'PREV FAILED'
                current_instrument = (current_instrument - 1) % len(instruments)
                action_text = f'PREV -> {instruments[current_instrument]}'
                last_action_time = time.time()
                play_pulse = False
                print('[GESTURE] swipe left -> previous track')

            else:
                # compute remaining cooldown for toggle (for UI)
                if recognizer.last_trigger.get('toggle'):
                    tleft = max(0, recognizer.cooldown - (time.time() - recognizer.last_trigger.get('toggle', 0)))
                    cooldown_indicator = f'Toggle cooldown: {tleft:.1f}s' if tleft > 0 else ''
                try:
                    cur = controller.currently_playing() if controller else None
                except Exception:
                    cur = None
                overlay_text = cur or ('(no track)' if controller and not controller.is_available() else '')

            # show overlays
            draw_overlay(out_frame, overlay_text, current_volume)
            now_t = time.time()
            if now_t - last_action_time < 1.0 and action_text:
                cv2.putText(out_frame, action_text, (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 220, 80), 3)
            draw_instrument_panel(out_frame, instruments, current_instrument, instrument_vols, play_pulse, now_t)
            if cooldown_indicator:
                cv2.putText(out_frame, cooldown_indicator, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)

            cv2.imshow('AirSwipe - Gesture Player', out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gesture-controlled music player')
    parser.add_argument('--debug', action='store_true', help='Run without audio controller (print gestures only)')
    args = parser.parse_args()
    main(debug=args.debug)
