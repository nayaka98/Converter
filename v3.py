# midi_to_bin.py
# Requires: mido, numpy, Pillow
# Usage: python midi_to_bin.py

import sys
import math
import numpy as np
import mido
import shlex
from multiprocessing import Pool, cpu_count

def note_to_freq(note):
    return 440.0 * 2**((note - 69) / 12.0)

def make_waveform(wave_type, freq, length_samples, sample_rate):
    t = np.arange(length_samples) / sample_rate
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    if wave_type == 'square':
        return np.sign(np.sin(2 * np.pi * freq * t))
    if wave_type == 'triangle':
        return 2 * np.arcsin(np.sin(2 * np.pi * freq * t)) / np.pi
    if wave_type == 'saw':
        # simple sawtooth
        return 2*(t*freq - np.floor(0.5 + t*freq))
    # default
    return np.sin(2 * np.pi * freq * t)

def apply_adsr(env_length, sample_rate, attack=0.01, decay=0.05, sustain_level=0.8, release=0.05):
    # env_length in seconds
    L = int(round(env_length * sample_rate))
    if L <= 0:
        return np.array([], dtype=float)
    a_s = int(round(min(attack, env_length) * sample_rate))
    d_s = int(round(min(decay, max(0.0, env_length - attack - release)) * sample_rate))
    r_s = int(round(min(release, env_length) * sample_rate))
    s_s = L - (a_s + d_s + r_s)
    if s_s < 0:
        # shorten sustain if total > length, distribute proportionally
        total = a_s + d_s + r_s
        if total == 0:
            a_s = d_s = r_s = 0
            s_s = L
        else:
            factor = L / total
            a_s = int(round(a_s * factor))
            d_s = int(round(d_s * factor))
            r_s = L - (a_s + d_s)
            s_s = 0
    env = np.zeros(L, dtype=float)
    idx = 0
    if a_s > 0:
        env[idx:idx+a_s] = np.linspace(0.0, 1.0, a_s, endpoint=False)
        idx += a_s
    if d_s > 0:
        env[idx:idx+d_s] = np.linspace(1.0, sustain_level, d_s, endpoint=False)
        idx += d_s
    if s_s > 0:
        env[idx:idx+s_s] = sustain_level
        idx += s_s
    if r_s > 0:
        env[idx:idx+r_s] = np.linspace(sustain_level, 0.0, r_s, endpoint=True)
    return env

def build_note_events(mid):
    # collect all events with absolute ticks
    events = []
    for i, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
    events.sort(key=lambda x: x[0])

    current_tempo = 500000  # default microseconds per beat (120 bpm)
    prev_tick = 0
    current_time = 0.0
    active_notes = {}  # note -> (start_time, velocity)
    note_items = []    # list of (note, start, end, velocity)

    for abs_tick, msg in events:
        delta_ticks = abs_tick - prev_tick
        if delta_ticks:
            dt = mido.tick2second(delta_ticks, mid.ticks_per_beat, current_tempo)
            current_time += dt
            prev_tick = abs_tick

        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
            continue

        if msg.type == 'note_on' and msg.velocity > 0:
            # start note
            # if same note already active, close previous one
            if msg.note in active_notes:
                s_start, s_vel = active_notes.pop(msg.note)
                note_items.append((msg.note, s_start, current_time, s_vel))
            active_notes[msg.note] = (current_time, msg.velocity)
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                s_start, s_vel = active_notes.pop(msg.note)
                note_items.append((msg.note, s_start, current_time, s_vel))
            # else: orphan note_off -> ignore

    # any notes still active -> close at end time (use current_time)
    for note, (s_start, s_vel) in active_notes.items():
        note_items.append((note, s_start, current_time, s_vel))

    return note_items

def process_midi_note(args):
    note, start, end, vel, wave_type, sample_rate = args
    start_idx = int(round(start * sample_rate))
    end_idx = int(round(end * sample_rate))
    length_samples = max(1, end_idx - start_idx)
    freq = note_to_freq(note)
    base = make_waveform(wave_type, freq, length_samples, sample_rate)
    amp = (vel / 127.0)
    dur_seconds = (end - start) if (end - start) > 0 else (1.0 / 1000.0)
    env = apply_adsr(dur_seconds, sample_rate,
                     attack=0.005, decay=0.02, sustain_level=0.85, release=0.02)
    if env.size != base.size:
        env = np.interp(np.linspace(0, 1, base.size), np.linspace(0,1,env.size), env)
    note_wave = base * env * amp
    return start_idx, length_samples, note_wave

def midi_to_bin(midi_path, out_bin, wave_type='sine', sample_rate=44100, max_amp=0.95, noise_power=0.0):
    mid = mido.MidiFile(midi_path)
    notes = build_note_events(mid)
    if not notes:
        raise SystemExit("No notes detected in MIDI.")

    total_time = max(end for (_, _, end, _) in notes)
    n_samples = int(math.ceil(total_time * sample_rate)) + 1
    buffer = np.zeros(n_samples, dtype=float)

    # Prepare arguments for multiprocessing
    note_args = [(note, start, end, vel, wave_type, sample_rate) for note, start, end, vel in notes]
    
    print(f"Processing {len(notes)} notes using multiprocessing...")
    with Pool() as p:
        results = p.map(process_midi_note, note_args)
        
    for start_idx, length_samples, note_wave in results:
        buffer[start_idx:start_idx+length_samples] += note_wave

    if noise_power > 0.0:
        noise = np.random.uniform(-1.0, 1.0, len(buffer)) * noise_power
        buffer += noise

    # normalize to avoid clipping
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = buffer / max_val * max_amp

    # convert to unsigned 8-bit (0..255)
    pcm_u8 = np.clip((buffer * 127.0) + 128.0, 0, 255).astype(np.uint8)

    with open(out_bin, "wb") as f:
        f.write(pcm_u8.tobytes())

    print(f"OK: ditulis {out_bin}  (durasi ≈ {total_time:.3f} s, sample_rate={sample_rate}, waveform={wave_type}, noise={noise_power})")

def process_image_column(args):
    x, col_pixels, freqs, samples_per_col, sample_rate, height = args
    col_audio = np.zeros(samples_per_col, dtype=float)
    t_col = np.arange(samples_per_col) / sample_rate
    
    for y in range(height):
        pixel_val = col_pixels[y]
        if pixel_val > 0:
            amp = (pixel_val / 255.0) * (1.0 / math.sqrt(height))
            freq = freqs[y]
            phase = 2 * np.pi * freq * x * (samples_per_col / sample_rate)
            wave = amp * np.sin(2 * np.pi * freq * t_col + phase)
            col_audio += wave
            
    return x, col_audio

def image_to_bin(image_path, out_bin, pixel_time_ms=20.0, sample_rate=44100, max_amp=0.95):
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow library is required for image processing. Install it with 'pip install Pillow'")
        return

    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
    except Exception as e:
        print(f"Error opening image: {e}")
        return
        
    width, height = img.size
    
    # Map rows to frequencies (bottom row = low freq, top row = high freq)
    # Let's use a logarithmic scale from e.g. 50 Hz to 10000 Hz
    min_freq = 50.0
    max_freq = 10000.0
    
    # Calculate frequencies for each row (0 is top, height-1 is bottom)
    freqs = np.zeros(height)
    for y in range(height):
        # y=0 -> max_freq, y=height-1 -> min_freq
        fraction = (height - 1 - y) / max(1, height - 1)
        # Logarithmic interpolation
        freqs[y] = min_freq * (max_freq / min_freq) ** fraction
        
    samples_per_col = int((pixel_time_ms / 1000.0) * sample_rate)
    if samples_per_col <= 0:
        samples_per_col = 1
        
    total_samples = width * samples_per_col
    buffer = np.zeros(total_samples, dtype=float)
    
    img_data = np.array(img) # shape: (height, width)
    
    print(f"Processing image {width}x{height} using multiprocessing...")
    
    # Prepare arguments for multiprocessing
    col_args = []
    for x in range(width):
        col_pixels = img_data[:, x]
        col_args.append((x, col_pixels, freqs, samples_per_col, sample_rate, height))
        
    with Pool() as p:
        results = p.map(process_image_column, col_args)
        
    for x, col_audio in results:
        start_idx = x * samples_per_col
        buffer[start_idx:start_idx+samples_per_col] = col_audio
        
    # Normalize
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = buffer / max_val * max_amp
        
    # Convert to 8-bit PCM
    pcm_u8 = np.clip((buffer * 127.0) + 128.0, 0, 255).astype(np.uint8)
    
    with open(out_bin, "wb") as f:
        f.write(pcm_u8.tobytes())
        
    total_time = total_samples / sample_rate
    print(f"OK: Image written to {out_bin} (duration ≈ {total_time:.3f} s, sample_rate={sample_rate})")

if __name__ == "__main__":
    print("MIDI to BIN Converter Terminal")
    print("Type 'help' to see all commands.")
    print("Type 'exit' or 'quit' to close.")
    
    while True:
        try:
            cmd_line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
            
        if not cmd_line.strip():
            continue
            
        args = shlex.split(cmd_line)
        cmd = args[0].lower()
        
        if cmd in ['exit', 'quit']:
            break
        elif cmd == 'help':
            print("Commands:")
            print("  process <input.mid> <output.bin> [noise=<value>] [wave=<type>]")
            print("      Convert MIDI to BIN audio.")
            print("      Options: noise=0.4 (adds white noise), wave=sine|square|triangle|saw")
            print("  image <input.png> <output.bin> <pixel_time>")
            print("      Convert Image to BIN audio (spectrogram/waterfall).")
            print("      Example: image 1.png 2.bin 20p (20ms per pixel column)")
            print("  help")
            print("      Show this help message.")
            print("  exit | quit")
            print("      Exit the program.")
        elif cmd == 'image':
            if len(args) < 4:
                print("Usage: image <input.png> <output.bin> <pixel_time>")
                print("Example: image 1.png 2.bin 20p")
                continue
            
            img_path = args[1]
            out_bin = args[2]
            p_time_str = args[3]
            
            pixel_time = 20.0
            if p_time_str.lower().endswith('p'):
                try:
                    pixel_time = float(p_time_str[:-1])
                except ValueError:
                    print(f"Invalid pixel time: {p_time_str}")
            else:
                try:
                    pixel_time = float(p_time_str)
                except ValueError:
                    print(f"Invalid pixel time: {p_time_str}")
                    
            try:
                image_to_bin(img_path, out_bin, pixel_time_ms=pixel_time)
            except Exception as e:
                print(f"Error processing image: {e}")
        elif cmd == 'process':
            if len(args) < 3:
                print("Usage: process <input.mid> <output.bin> [noise=<value>] [wave=<type>]")
                continue
                
            midi_path = args[1]
            out_bin = args[2]
            
            noise_val = 0.0
            wave_val = 'sine'
            
            for arg in args[3:]:
                if arg.startswith('noise='):
                    try:
                        noise_val = float(arg.split('=')[1])
                    except ValueError:
                        print(f"Invalid noise value: {arg}")
                elif arg.startswith('wave='):
                    wave_val = arg.split('=')[1]
            
            try:
                midi_to_bin(midi_path, out_bin, wave_type=wave_val, noise_power=noise_val)
            except Exception as e:
                print(f"Error processing: {e}")
        else:
            print(f"Unknown command: {cmd}")

