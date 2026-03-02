# midi_to_bin.py
# Requires: mido, numpy
# Usage: python midi_to_bin.py /sdcard/termuxdownload/1.mid /sdcard/termuxdownload/1.bin

import sys
import math
import numpy as np
import mido

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

def midi_to_bin(midi_path, out_bin, wave_type='sine', sample_rate=44100, max_amp=0.95):
    mid = mido.MidiFile(midi_path)
    notes = build_note_events(mid)
    if not notes:
        raise SystemExit("No notes detected in MIDI.")

    total_time = max(end for (_, _, end, _) in notes)
    n_samples = int(math.ceil(total_time * sample_rate)) + 1
    buffer = np.zeros(n_samples, dtype=float)

    for note, start, end, vel in notes:
        start_idx = int(round(start * sample_rate))
        end_idx = int(round(end * sample_rate))
        length_samples = max(1, end_idx - start_idx)
        freq = note_to_freq(note)
        base = make_waveform(wave_type, freq, length_samples, sample_rate)
        # apply velocity (0-127)
        amp = (vel / 127.0)
        # envelope
        dur_seconds = (end - start) if (end - start) > 0 else (1.0 / 1000.0)
        env = apply_adsr(dur_seconds, sample_rate,
                         attack=0.005, decay=0.02, sustain_level=0.85, release=0.02)
        if env.size != base.size:
            # stretch/shrink env to fit
            env = np.interp(np.linspace(0, 1, base.size), np.linspace(0,1,env.size), env)
        note_wave = base * env * amp
        buffer[start_idx:start_idx+length_samples] += note_wave

    # normalize to avoid clipping
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = buffer / max_val * max_amp

    # convert to unsigned 8-bit (0..255)
    pcm_u8 = np.clip((buffer * 127.0) + 128.0, 0, 255).astype(np.uint8)

    with open(out_bin, "wb") as f:
        f.write(pcm_u8.tobytes())

    print(f"OK: ditulis {out_bin}  (durasi ≈ {total_time:.3f} s, sample_rate={sample_rate}, waveform={wave_type})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python midi_to_bin.py <input.mid> <output.bin> [waveform]")
        print("Waveform options: sine (default), square, triangle, saw")
        sys.exit(1)
    midi_path = sys.argv[1]
    out_bin = sys.argv[2]
    wave = sys.argv[3] if len(sys.argv) > 3 else 'sine'
    midi_to_bin(midi_path, out_bin, wave_type=wave)
