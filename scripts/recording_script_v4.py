#!/usr/bin/env python3
"""
Muse S EEG + Video Recording Script (v4 with Auto-Reconnect)

This script simultaneously records:
1. Raw EEG data from Muse S headband via Muse-LSL (Lab Streaming Layer)
2. Video from webcam/camera using OpenCV

v4 Features:
- Auto-reconnect on connection loss
- Connection loss detection with terminal warnings
- Creates new CSV files (_2.csv, _3.csv) on reconnect
- Real-time frequency monitoring during recording

Requirements:
- Muse S headband paired and connected
- Muse-LSL running (or uvicmuse) to stream EEG data
- OpenCV for video capture
- pylsl for LSL stream handling

Usage:
    python recording_script_v4.py --duration 1800
    python recording_script_v4.py --nocamera --duration 1800  # EEG only, no video

Author: Generated for Muse S TikTok EEG Study
"""

import argparse
import time
import threading
import queue
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("WARNING: pynput not installed. Keypress logging will not work.")
    print("Install with: pip install pynput")

try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False
    print("WARNING: pylsl not installed. Install with: pip install pylsl")
    print("EEG recording will not work without pylsl.")
except RuntimeError as e:
    HAS_PYLSL = False
    error_msg = str(e)
    print("ERROR: pylsl requires the LSL binary library (liblsl) to be installed separately.")
    print("\nInstallation options:")
    print("\n1. Try Conda (often works better on macOS):")
    print("   conda install -c conda-forge liblsl")
    print("   # Then set: export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH")
    print("\n2. Manual download (if Homebrew fails):")
    print("   Download from: https://github.com/sccn/liblsl/releases")
    print("   Extract and set: export PYLSL_LIB=/path/to/liblsl.dylib")
    print("\n3. If Homebrew installed but can't find it:")
    brew_prefix = None
    try:
        import subprocess
        result = subprocess.run(['brew', '--prefix'], capture_output=True, text=True)
        if result.returncode == 0:
            brew_prefix = result.stdout.strip()
            print(f"\n   Try: export DYLD_LIBRARY_PATH={brew_prefix}/lib:$DYLD_LIBRARY_PATH")
            print(f"   Or: export PYLSL_LIB={brew_prefix}/lib/liblsl.dylib")
    except:
        pass
    print(f"\nOriginal error: {error_msg}")
    raise


class KeypressLogger:
    """Logs keypress events with timestamps for synchronization with EEG."""
    
    def __init__(self, debug=False):
        self.keypress_a_queue = queue.Queue()
        self.keypress_b_queue = queue.Queue()
        self.is_active = False
        self.listener = None
        self.debug = debug
        self.keypress_a_count = 0
        self.keypress_b_count = 0
        
    def on_key_press(self, key):
        """Callback for key press events."""
        try:
            timestamp = time.time()
            if key.char == 'a' or key.char == 'A':
                self.keypress_a_queue.put(timestamp)
                self.keypress_a_count += 1
                if self.debug:
                    print(f"[DEBUG] Key 'A' pressed at {timestamp:.4f} (total: {self.keypress_a_count})")
            elif key.char == 'b' or key.char == 'B':
                self.keypress_b_queue.put(timestamp)
                self.keypress_b_count += 1
                if self.debug:
                    print(f"[DEBUG] Key 'B' pressed at {timestamp:.4f} (total: {self.keypress_b_count})")
        except AttributeError:
            # Special keys (ctrl, alt, etc.) don't have char attribute
            if self.debug:
                print(f"[DEBUG] Special key pressed: {key}")
            pass
    
    def start(self):
        """Start keypress monitoring."""
        if not HAS_PYNPUT:
            print("WARNING: pynput not available. Keypress logging disabled.")
            return
        
        self.is_active = True
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        if self.debug:
            print("[DEBUG] Keypress listener started successfully")
        print("Keypress logging started. Press 'A' or 'B' keys to mark events.")
    
    def stop(self):
        """Stop keypress monitoring."""
        self.is_active = False
        if self.listener:
            self.listener.stop()
        print("Keypress logging stopped.")
    
    def get_recent_keypresses(self, current_time, key_type='a', tolerance=0.100, time_offset=None):
        """Get keypresses within tolerance of current_time (default 100ms).
        
        Args:
            current_time: LSL timestamp to match against
            key_type: 'a' or 'b' 
            tolerance: Time window in seconds
            time_offset: Offset to convert system time keypresses to LSL time
        """
        keypresses = []
        temp_queue = queue.Queue()
        
        # Select the appropriate queue
        if key_type.lower() == 'a':
            target_queue = self.keypress_a_queue
            key_name = 'A'
        elif key_type.lower() == 'b':
            target_queue = self.keypress_b_queue
            key_name = 'B'
        else:
            return False
        
        queue_size_before = target_queue.qsize()
        
        # Check all queued keypresses
        while not target_queue.empty():
            try:
                keypress_system_time = target_queue.get_nowait()
                
                # Convert keypress system time to LSL time if offset available
                if time_offset is not None:
                    keypress_lsl_time = keypress_system_time - time_offset
                    time_diff = abs(current_time - keypress_lsl_time)
                    
                    if time_diff <= tolerance:
                        keypresses.append(keypress_system_time)
                        if self.debug:
                            print(f"[DEBUG] Key {key_name} matched! LSL time diff: {time_diff:.6f}s (tolerance: {tolerance:.6f}s)")
                            print(f"[DEBUG] Keypress: sys={keypress_system_time:.6f}, lsl={keypress_lsl_time:.6f}, sample_lsl={current_time:.6f}")
                    elif keypress_lsl_time > current_time - tolerance:
                        # Keep future keypresses for next check
                        temp_queue.put(keypress_system_time)
                    else:
                        # Discard old keypresses
                        if self.debug:
                            print(f"[DEBUG] Discarded old {key_name} keypress: {time_diff:.6f}s ago (LSL time)")
                else:
                    # Fallback to system time comparison if no offset available
                    time_diff = abs(current_time - keypress_system_time)
                    if time_diff <= tolerance:
                        keypresses.append(keypress_system_time)
                        if self.debug:
                            print(f"[DEBUG] Key {key_name} matched! Time diff: {time_diff:.6f}s (tolerance: {tolerance:.6f}s)")
                    elif keypress_system_time > current_time - tolerance:
                        temp_queue.put(keypress_system_time)
                    else:
                        if self.debug:
                            print(f"[DEBUG] Discarded old {key_name} keypress: {time_diff:.6f}s ago")
            except queue.Empty:
                break
        
        # Put back future keypresses
        while not temp_queue.empty():
            target_queue.put(temp_queue.get())
        
        if self.debug and queue_size_before > 0:
            print(f"[DEBUG] Checked {queue_size_before} queued {key_name} keypresses, found {len(keypresses)} matches")
        
        return len(keypresses) > 0


class EEGRecorderWithReconnect:
    """Records EEG data from Muse S via LSL stream with auto-reconnect capability."""
    
    def __init__(self, recording_folder, base_filename, stream_name="MuseS", keypress_logger=None, debug=False):
        self.recording_folder = Path(recording_folder)
        self.base_filename = base_filename
        self.stream_name = stream_name
        self.is_recording = False
        self.data_queue = queue.Queue()
        self.thread = None
        self.inlet = None
        self.sample_count = 0
        self.total_sample_count = 0  # Across all CSV files
        self.keypress_logger = keypress_logger
        self.debug = debug
        self.time_offset = None
        self.last_sample_time = None  # For connection loss detection
        self.connection_lost = False
        self.csv_file_count = 1  # Track number of CSV files
        self.csv_files = []  # List of all CSV files created
        self.current_output_file = None
        self._lock = threading.Lock()
        
    def get_current_output_file(self):
        """Get the current output file path."""
        if self.csv_file_count == 1:
            filename = f"{self.base_filename}.csv"
        else:
            filename = f"{self.base_filename}_{self.csv_file_count}.csv"
        return self.recording_folder / filename
        
    def find_stream(self, timeout=10.0, quiet=False):
        """Find the Muse S LSL stream."""
        if not quiet:
            print(f"Looking for LSL stream: {self.stream_name}...")
        try:
            # First try exact name match
            streams = pylsl.resolve_byprop('name', self.stream_name, timeout=timeout)
            
            if len(streams) == 0:
                # Try to find any Muse stream
                if not quiet:
                    print(f"Stream '{self.stream_name}' not found. Searching for any Muse stream...")
                all_streams = pylsl.resolve_streams(wait_time=timeout)
                muse_streams = [s for s in all_streams if 'Muse' in s.name() or 'muse' in s.name().lower()]
                if len(muse_streams) == 0:
                    return False
                streams = muse_streams
            
            stream_info = streams[0]
            if not quiet:
                print(f"Found stream: {stream_info.name()}")
                print(f"  Type: {stream_info.type()}")
                print(f"  Channels: {stream_info.channel_count()}")
                print(f"  Sampling rate: {stream_info.nominal_srate()} Hz")
            
            self.inlet = pylsl.StreamInlet(stream_info)
            return True
        except Exception as e:
            if not quiet:
                print(f"Error finding LSL stream: {e}")
            return False
    
    def record_loop(self):
        """Main recording loop (runs in separate thread)."""
        if self.inlet is None:
            print("ERROR: LSL stream not initialized")
            return
        
        # Get stream info
        info = self.inlet.info()
        n_channels = info.channel_count()
        fs = info.nominal_srate()
        
        # Channel names
        channel_names = []
        ch = info.desc().child("channels").child("channel")
        for i in range(n_channels):
            channel_names.append(ch.child_value("label") or f"CH{i+1}")
            ch = ch.next_sibling()
        
        if not any(channel_names) or len(channel_names) != n_channels:
            default_names = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX_RIGHT', 'AUX_LEFT']
            channel_names = default_names[:n_channels]
        
        print(f"Recording from {n_channels} channels: {channel_names}")
        print(f"Sampling rate: {fs} Hz")
        
        # Open CSV file for writing
        self.current_output_file = self.get_current_output_file()
        self.csv_files.append(str(self.current_output_file))
        
        with open(self.current_output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['timestamp', 'lsl_timestamp'] + channel_names + ['keypress_A', 'keypress_B']
            writer.writerow(header)
            
            while self.is_recording:
                try:
                    sample, lsl_timestamp = self.inlet.pull_sample(timeout=1.0)
                    
                    if sample:
                        system_time = time.time()
                        
                        with self._lock:
                            self.last_sample_time = system_time
                            self.connection_lost = False
                        
                        if self.time_offset is None:
                            self.time_offset = system_time - lsl_timestamp
                        
                        keypress_a_flag = 0
                        keypress_b_flag = 0
                        if self.keypress_logger and self.keypress_logger.is_active:
                            if self.keypress_logger.get_recent_keypresses(lsl_timestamp, 'a', time_offset=self.time_offset):
                                keypress_a_flag = 1
                            if self.keypress_logger.get_recent_keypresses(lsl_timestamp, 'b', time_offset=self.time_offset):
                                keypress_b_flag = 1
                        
                        row = [system_time, lsl_timestamp] + list(sample) + [keypress_a_flag, keypress_b_flag]
                        writer.writerow(row)
                        
                        self.sample_count += 1
                        self.total_sample_count += 1
                        
                except Exception as e:
                    if self.is_recording:
                        with self._lock:
                            self.connection_lost = True
                        if self.debug:
                            print(f"Error in EEG recording loop: {e}")
                    break
        
        print(f"EEG recording stopped. Samples in this file: {self.sample_count}")
    
    def is_connection_lost(self, timeout=5.0):
        """Check if connection has been lost (no samples for timeout seconds)."""
        with self._lock:
            if self.last_sample_time is None:
                return False
            return (time.time() - self.last_sample_time) > timeout or self.connection_lost
    
    def get_time_since_last_sample(self):
        """Get time since last sample was received."""
        with self._lock:
            if self.last_sample_time is None:
                return 0
            return time.time() - self.last_sample_time
    
    def start(self):
        """Start recording."""
        if not HAS_PYLSL:
            raise RuntimeError("pylsl not available. Cannot record EEG.")
        
        if not self.find_stream():
            raise RuntimeError("Could not find LSL stream")
        
        self.is_recording = True
        self.sample_count = 0
        self.thread = threading.Thread(target=self.record_loop, daemon=True)
        self.thread.start()
        print("EEG recording started...")
    
    def stop(self):
        """Stop recording."""
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print(f"EEG recording completed. Total samples: {self.total_sample_count}")
        return self.total_sample_count
    
    def reconnect(self, muselsl_process=None):
        """Attempt to reconnect by restarting muselsl with any available device.
        
        Args:
            muselsl_process: The current muselsl subprocess to terminate
            
        Returns:
            tuple: (success: bool, new_muselsl_process: subprocess or None)
        """
        # Stop current recording thread
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Kill old muselsl if provided
        if muselsl_process:
            try:
                muselsl_process.terminate()
                muselsl_process.wait(timeout=2)
            except:
                try:
                    muselsl_process.kill()
                except:
                    pass
        
        # Scan for any Muse device
        devices = scan_muse_devices()
        if not devices:
            return False, None
        
        # Use the first device found
        device = devices[0]
        device_name = device['name']
        
        # Start new muselsl stream
        new_process = start_muselsl_stream(device_name)
        if not new_process:
            return False, None
        
        # Wait for LSL stream to become available
        if not wait_for_lsl_stream(device_name, timeout=15):
            # Try with generic "Muse" name
            time.sleep(2)
        
        # Increment file counter for new CSV
        self.csv_file_count += 1
        self.sample_count = 0
        self.inlet = None
        self.time_offset = None
        
        # Try to find the new stream
        if self.find_stream(timeout=10.0, quiet=False):
            self.is_recording = True
            with self._lock:
                self.connection_lost = False
                self.last_sample_time = None
            self.thread = threading.Thread(target=self.record_loop, daemon=True)
            self.thread.start()
            return True, new_process
        
        return False, new_process


# Keep the old class name as an alias for backwards compatibility
EEGRecorder = EEGRecorderWithReconnect


class VideoRecorder:
    """Records video from webcam/camera."""
    
    def __init__(self, output_file, camera_index=0, fps=30, resolution=(1280, 720)):
        self.output_file = output_file
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.is_recording = False
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None
        self.actual_fps = fps
        
    def start(self):
        """Start video recording."""
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Use requested FPS or camera FPS, whichever is more reliable
        # For real-time recording, use the requested FPS
        self.actual_fps = self.fps if self.fps > 0 else camera_fps
        
        print(f"Video recording settings:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  Target FPS: {self.actual_fps}")
        print(f"  Camera reported FPS: {camera_fps}")
        
        # Use H.264 codec (avc1) for better macOS compatibility and accurate playback
        # Fallback to mp4v if avc1 doesn't work
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        self.writer = cv2.VideoWriter(
            str(self.output_file),
            fourcc,
            self.actual_fps,
            (actual_width, actual_height)
        )
        
        # If avc1 fails, try mp4v
        if not self.writer.isOpened():
            print("Warning: avc1 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(self.output_file),
                fourcc,
                self.actual_fps,
                (actual_width, actual_height)
            )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create video writer for {self.output_file}")
        
        self.is_recording = True
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        print("Video recording started...")
    
    def record_frame(self, frame=None):
        """Record a single frame (call in loop).
        
        Parameters:
        -----------
        frame : numpy array, optional
            If provided, use this frame instead of reading from camera
        """
        if not self.is_recording:
            return False, None
        
        # Read frame if not provided
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Could not read frame from camera")
                return False, None
        else:
            ret = True
        
        # Write frame
        self.writer.write(frame)
        self.frame_count += 1
        
        return True, frame
    
    def stop(self):
        """Stop video recording."""
        self.is_recording = False
        if self.writer:
            self.writer.release()
        if self.cap:
            self.cap.release()
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Calculate actual FPS from recording
        actual_recorded_fps = self.frame_count / duration if duration > 0 else 0
        
        print(f"Video recording completed.")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Actual FPS: {actual_recorded_fps:.2f}")
        print(f"  Video file FPS setting: {self.actual_fps}")
        
        # Warn if there's a mismatch
        if abs(actual_recorded_fps - self.actual_fps) > 2.0:
            print(f"  WARNING: Recorded FPS ({actual_recorded_fps:.2f}) differs from video FPS setting ({self.actual_fps})")
            print(f"  This may cause playback speed issues. Consider using --fps {int(actual_recorded_fps)}")


def scan_muse_devices():
    """Scan for available Muse devices using muselsl."""
    print("Scanning for Muse devices...")
    try:
        result = subprocess.run(
            ['muselsl', 'list'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            print(f"Error scanning for devices: {result.stderr}")
            return []
        
        # Parse output
        devices = []
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Found device' in line:
                # Parse: "Found device MuseS-4646, MAC Address C4CD9D92-74EC-861B-CC7B-E67E126857D5"
                parts = line.split(',')
                if len(parts) >= 2:
                    name_part = parts[0].split('Found device')[1].strip()
                    mac_part = parts[1].split('MAC Address')[1].strip()
                    devices.append({
                        'name': name_part,
                        'mac': mac_part
                    })
        
        return devices
    except subprocess.TimeoutExpired:
        print("Scan timed out. Make sure your Muse headband is powered on and in range.")
        return []
    except FileNotFoundError:
        print("muselsl not found. Please install with: pip install muselsl")
        return []
    except Exception as e:
        print(f"Error scanning for devices: {e}")
        return []


def select_muse_device():
    """Present menu to select a Muse device and return device info."""
    devices = scan_muse_devices()
    
    if len(devices) == 0:
        print("\nNo Muse devices found.")
        print("Make sure:")
        print("  - Your Muse headband is powered on")
        print("  - Bluetooth is enabled on your Mac")
        print("  - The headband is in range")
        return None
    
    print(f"\nFound {len(devices)} device(s):")
    print("-" * 60)
    for i, device in enumerate(devices, 1):
        print(f"  {i}. {device['name']} (MAC: {device['mac']})")
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"\nSelect device (1-{len(devices)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                selected = devices[idx]
                print(f"\nSelected: {selected['name']}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(devices)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def scan_cameras(max_cameras=10):
    """Scan for available cameras and return list of working camera indices."""
    available_cameras = []
    
    print("\nScanning for available cameras...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it works
            ret, frame = cap.read()
            if ret:
                # Get camera name/info if available
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'backend': backend,
                    'resolution': (width, height),
                    'fps': fps
                })
            cap.release()
    
    return available_cameras


def select_camera(auto_select=True):
    """Select a camera - auto-selects non-default camera if multiple found."""
    cameras = scan_cameras()
    
    if len(cameras) == 0:
        print("\nNo cameras found. Using default camera (index 0).")
        return 0
    
    print(f"\nFound {len(cameras)} camera(s):")
    print("-" * 60)
    for i, cam in enumerate(cameras, 1):
        print(f"  {i}. Camera {cam['index']} - {cam['resolution'][0]}x{cam['resolution'][1]} @ {cam['fps']:.1f} fps ({cam['backend']})")
    print("-" * 60)
    
    if auto_select:
        # Auto-select: prefer camera 0 (external camera) if available
        # If camera 0 exists, use it; otherwise use the first available camera
        camera_0 = [cam for cam in cameras if cam['index'] == 0]
        if len(camera_0) > 0:
            selected = camera_0[0]
            print(f"\nAuto-selected: Camera {selected['index']} (default/external camera)")
            return selected['index']
        else:
            # Camera 0 not available, use first available
            selected = cameras[0]
            print(f"\nAuto-selected: Camera {selected['index']} (camera 0 not available)")
            return selected['index']
    else:
        # Interactive selection (disabled for now)
        while True:
            try:
                choice = input(f"\nSelect camera (1-{len(cameras)}) or 'q' to use default (0): ").strip()
                
                if choice.lower() == 'q' or choice == '':
                    print("\nUsing default camera (index 0)")
                    return 0
                
                idx = int(choice) - 1
                if 0 <= idx < len(cameras):
                    selected = cameras[idx]
                    print(f"\nSelected: Camera {selected['index']}")
                    return selected['index']
                else:
                    print(f"Please enter a number between 1 and {len(cameras)}")
            except ValueError:
                print("Please enter a valid number, 'q' to use default, or press Enter")
            except KeyboardInterrupt:
                print("\nUsing default camera (index 0)")
                return 0


def start_muselsl_stream(device_name, include_acc=False, include_gyro=False):
    """Start muselsl stream as a subprocess."""
    print(f"\nStarting muselsl stream for {device_name}...")
    
    cmd = ['muselsl', 'stream', '--name', device_name]
    if include_acc:
        cmd.append('--acc')
    if include_gyro:
        cmd.append('--gyro')
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment to see if it starts successfully
        time.sleep(2)
        
        if process.poll() is not None:
            # Process ended, check for errors
            stdout, stderr = process.communicate()
            print(f"Error starting muselsl: {stderr}")
            return None
        
        print("muselsl stream started successfully!")
        print("(Keep this terminal open while recording)")
        return process
        
    except FileNotFoundError:
        print("ERROR: muselsl not found. Install with: pip install muselsl")
        return None
    except Exception as e:
        print(f"Error starting muselsl: {e}")
        return None


def wait_for_lsl_stream(stream_name="MuseS", timeout=30):
    """Wait for LSL stream to become available."""
    print(f"\nWaiting for LSL stream '{stream_name}' to become available...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            streams = pylsl.resolve_byprop('name', stream_name, timeout=2.0)
            if len(streams) > 0:
                print(f"✓ LSL stream '{stream_name}' is ready!")
                return True
            
            # Also check for any Muse stream
            all_streams = pylsl.resolve_streams(wait_time=2.0)
            muse_streams = [s for s in all_streams if 'Muse' in s.name() or 'muse' in s.name().lower()]
            if len(muse_streams) > 0:
                print(f"✓ Found Muse stream: {muse_streams[0].name()}")
                return True
                
        except Exception:
            pass
        
        print(".", end="", flush=True)
        time.sleep(1)
    
    print(f"\n✗ Timeout: LSL stream not found after {timeout} seconds")
    return False


def record_synchronized(eeg_recorder, video_recorder, duration=None, show_video=True, keypress_logger=None):
    """
    Record EEG and video simultaneously.
    
    Parameters:
    -----------
    eeg_recorder : EEGRecorder
        EEG recorder instance
    video_recorder : VideoRecorder
        Video recorder instance
    duration : float, optional
        Recording duration in seconds. If None, record until interrupted.
    show_video : bool
        Whether to display video preview window
    keypress_logger : KeypressLogger, optional
        Keypress logger instance for event marking
    """
    # Start both recorders
    print("\n" + "="*60)
    print("Starting synchronized recording...")
    print("="*60)
    
    try:
        # Start keypress logging if available
        if keypress_logger:
            keypress_logger.start()
        
        eeg_recorder.start()
        time.sleep(0.5)  # Small delay to ensure EEG stream is ready
        
        video_recorder.start()
        
        # Recording loop
        start_time = time.time()
        last_status_time = start_time
        last_frame_time = start_time
        target_frame_interval = 1.0 / video_recorder.actual_fps  # Time between frames
        
        print("\nRecording... (Press 'q' to stop, or Ctrl+C)")
        print("="*60)
        
        while True:
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"\nRecording duration ({duration}s) reached. Stopping...")
                break
            
            # Control frame rate - wait if we're recording too fast
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time
            
            if time_since_last_frame < target_frame_interval:
                # Wait until it's time for the next frame
                time.sleep(target_frame_interval - time_since_last_frame)
            
            # Read frame once (use for both recording and preview)
            ret, frame = video_recorder.cap.read()
            if not ret:
                print("Video recording error. Stopping...")
                break
            
            # Record video frame
            success, recorded_frame = video_recorder.record_frame(frame)
            if not success:
                print("Video recording error. Stopping...")
                break
            
            last_frame_time = time.time()
            
            # Show video preview if requested
            if show_video:
                # Use the same frame we just recorded
                display_frame = frame.copy()
                # Add timestamp overlay
                elapsed = time.time() - start_time
                timestamp_text = f"Time: {elapsed:.2f}s | Frames: {video_recorder.frame_count}"
                cv2.putText(display_frame, timestamp_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Recording - Press Q to stop', display_frame)
                
                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n'q' pressed. Stopping recording...")
                    break
            
            # Print status every 10 seconds
            current_time = time.time()
            if current_time - last_status_time >= 10.0:
                elapsed = current_time - start_time
                # Calculate real-time frequency
                actual_freq = eeg_recorder.sample_count / elapsed if elapsed > 0 else 0
                expected_freq = 256.0
                freq_ok = abs(actual_freq - expected_freq) <= expected_freq * 0.20
                freq_indicator = "✓" if freq_ok else "✗"
                freq_color = "\033[92m" if freq_ok else "\033[91m"  # Green or Red
                reset_color = "\033[0m"
                print(f"Recording... {elapsed:.1f}s | "
                      f"EEG: {eeg_recorder.sample_count} | "
                      f"Video: {video_recorder.frame_count} | "
                      f"Freq: {freq_color}{actual_freq:.1f}Hz {freq_indicator}{reset_color}")
                last_status_time = current_time
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping recording...")
    except Exception as e:
        print(f"\n\nError during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all recorders
        print("\nStopping recorders...")
        video_recorder.stop()
        total_samples = eeg_recorder.stop()
        
        if keypress_logger:
            keypress_logger.stop()
        
        if show_video:
            cv2.destroyAllWindows()
        
        # Calculate and display EEG frequency analysis
        end_time = time.time()
        actual_duration = end_time - start_time
        if actual_duration > 0 and total_samples > 0:
            actual_freq = total_samples / actual_duration
            expected_freq = 256.0
            freq_tolerance = 0.20  # ±20%
            freq_min = expected_freq * (1 - freq_tolerance)
            freq_max = expected_freq * (1 + freq_tolerance)
            
            # Color coding for frequency check
            if freq_min <= actual_freq <= freq_max:
                freq_status = f"\033[92m✓ GOOD\033[0m"  # Green
            else:
                freq_status = f"\033[91m✗ BAD\033[0m"   # Red
            
            print(f"\nEEG Frequency Check: {actual_freq:.2f} Hz (expected: {expected_freq} Hz ±20%) - {freq_status}")
            print(f"Samples: {total_samples}, Duration: {actual_duration:.2f}s")
        
        print("\n" + "="*60)
        print("Recording completed!")
        print("="*60)


def record_eeg_only(eeg_recorder, duration=None, keypress_logger=None, muselsl_process=None):
    """
    Record EEG data only (no video) with auto-reconnect capability.
    
    Parameters:
    -----------
    eeg_recorder : EEGRecorderWithReconnect
        EEG recorder instance with reconnect capability
    duration : float, optional
        Recording duration in seconds. If None, record until interrupted.
    keypress_logger : KeypressLogger, optional
        Keypress logger instance for event marking
    muselsl_process : subprocess, optional
        The muselsl subprocess for reconnection management
    """
    print("\n" + "="*60)
    print("Starting EEG-only recording (no camera)...")
    print("="*60)
    
    start_time = time.time()
    last_sample_count = 0
    no_data_warning_shown = False
    reconnect_attempts = 0
    max_reconnect_wait = 30  # Max seconds to wait between reconnect attempts
    current_muselsl = muselsl_process  # Track the current muselsl process
    
    # Color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    try:
        # Start keypress logging if available
        if keypress_logger:
            keypress_logger.start()
        
        eeg_recorder.start()
        
        # Recording loop
        last_status_time = start_time
        
        print("\nRecording... (Press Ctrl+C to stop)")
        print("="*60)
        
        while True:
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"\nRecording duration ({duration}s) reached. Stopping...")
                break
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check for connection loss (no new samples for 3+ seconds)
            time_since_last = eeg_recorder.get_time_since_last_sample()
            current_sample_count = eeg_recorder.total_sample_count
            
            if time_since_last > 3.0 and current_sample_count == last_sample_count and current_sample_count > 0:
                if not no_data_warning_shown:
                    print(f"\n{RED}⚠️  NO DATA - Connection may be lost!{RESET}")
                    no_data_warning_shown = True
                
                # Check if we should attempt reconnect (after 5 seconds of no data)
                if time_since_last > 5.0 and eeg_recorder.is_connection_lost():
                    reconnect_attempts += 1
                    print(f"{YELLOW}reconnect.. (attempt {reconnect_attempts}) - scanning for devices...{RESET}")
                    
                    success, new_process = eeg_recorder.reconnect(current_muselsl)
                    if success:
                        if new_process:
                            current_muselsl = new_process
                        print(f"{GREEN}✓ Reconnected! Recording to new file: {eeg_recorder.current_output_file.name}{RESET}")
                        no_data_warning_shown = False
                        last_sample_count = eeg_recorder.total_sample_count
                        reconnect_attempts = 0  # Reset attempts on success
                    else:
                        # Wait before next attempt (exponential backoff up to max)
                        wait_time = min(2 ** reconnect_attempts, max_reconnect_wait)
                        time.sleep(min(wait_time, 2))  # Sleep max 2s per iteration to stay responsive
            else:
                if no_data_warning_shown and current_sample_count > last_sample_count:
                    print(f"{GREEN}✓ Data flowing again{RESET}")
                    no_data_warning_shown = False
                last_sample_count = current_sample_count
            
            # Print status every 10 seconds
            if current_time - last_status_time >= 10.0:
                # Calculate real-time frequency using total samples
                actual_freq = eeg_recorder.total_sample_count / elapsed if elapsed > 0 else 0
                expected_freq = 256.0
                freq_ok = abs(actual_freq - expected_freq) <= expected_freq * 0.20
                freq_indicator = "✓" if freq_ok else "✗"
                freq_color = GREEN if freq_ok else RED
                
                # Show file number if we've reconnected
                file_info = ""
                if eeg_recorder.csv_file_count > 1:
                    file_info = f" | File #{eeg_recorder.csv_file_count}"
                
                print(f"Recording... {elapsed:.1f}s | EEG: {eeg_recorder.total_sample_count} | "
                      f"Freq: {freq_color}{actual_freq:.1f}Hz {freq_indicator}{RESET}{file_info}")
                last_status_time = current_time
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping recording...")
    except Exception as e:
        print(f"\n\nError during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop EEG recorder
        print("\nStopping recorder...")
        total_samples = eeg_recorder.stop()
        
        if keypress_logger:
            keypress_logger.stop()
        
        # Calculate and display EEG frequency analysis
        end_time = time.time()
        actual_duration = end_time - start_time
        if actual_duration > 0 and total_samples > 0:
            actual_freq = total_samples / actual_duration
            expected_freq = 256.0
            freq_tolerance = 0.20  # ±20%
            freq_min = expected_freq * (1 - freq_tolerance)
            freq_max = expected_freq * (1 + freq_tolerance)
            
            # Color coding for frequency check
            if freq_min <= actual_freq <= freq_max:
                freq_status = f"{GREEN}✓ GOOD{RESET}"
            else:
                freq_status = f"{RED}✗ BAD{RESET}"
            
            print(f"\nEEG Frequency Check: {actual_freq:.2f} Hz (expected: {expected_freq} Hz ±20%) - {freq_status}")
            print(f"Total samples: {total_samples}, Duration: {actual_duration:.2f}s")
        
        # List all CSV files created
        if hasattr(eeg_recorder, 'csv_files') and len(eeg_recorder.csv_files) > 0:
            print(f"\nCSV files created ({len(eeg_recorder.csv_files)}):")
            for csv_file in eeg_recorder.csv_files:
                print(f"  - {csv_file}")
        
        print("\n" + "="*60)
        print("Recording completed!")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Record Muse S EEG + Video simultaneously',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record for 30 minutes (1800 seconds)
  python recording_script.py --duration 1800
  
  # Record with custom output directory
  python recording_script.py --output_dir ./my_recordings --duration 600
  
  # Record without video preview
  python recording_script.py --no-preview --duration 1200
        """
    )
    
    # Default recordings folder is ../recordings relative to this script's location
    script_dir = Path(__file__).resolve().parent
    default_recordings_dir = script_dir.parent / 'recordings'
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=str(default_recordings_dir),
        help=f'Output directory for recordings (default: {default_recordings_dir})'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Recording duration in seconds (default: record until interrupted)'
    )
    
    parser.add_argument(
        '--stream_name', '-s',
        type=str,
        default='MuseS',
        help='LSL stream name to look for (default: MuseS)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video FPS (default: 30)'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        default='1280x720',
        help='Video resolution WIDTHxHEIGHT (default: 1280x720)'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable video preview window'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for keypress detection'
    )
    
    parser.add_argument(
        '--auto-scan',
        action='store_true',
        default=True,
        help='Automatically scan for Muse devices and show menu (default: True)'
    )
    
    parser.add_argument(
        '--no-auto-scan',
        dest='auto_scan',
        action='store_false',
        help='Disable automatic device scanning (use --stream_name instead)'
    )
    
    parser.add_argument(
        '--include-acc',
        action='store_true',
        help='Include accelerometer data in stream'
    )
    
    parser.add_argument(
        '--include-gyro',
        action='store_true',
        help='Include gyroscope data in stream'
    )
    
    parser.add_argument(
        '--nocamera',
        action='store_true',
        help='Skip camera/video recording entirely (EEG only, saves storage)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_PYLSL:
        print("ERROR: pylsl is required for EEG recording.")
        print("Install with: pip install pylsl")
        print("\nAlternatively, you can use uvicmuse to stream EEG data separately.")
        return 1
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Warning: Invalid resolution format '{args.resolution}'. Using default 1280x720")
        resolution = (1280, 720)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subfolder for this recording session
    recording_folder = output_dir / f"eeg_{timestamp}"
    recording_folder.mkdir(parents=True, exist_ok=True)
    
    eeg_file = recording_folder / f"eeg_{timestamp}.csv"
    video_file = recording_folder / f"video_{timestamp}.mp4"
    
    print("="*60)
    print("Muse S EEG + Video Recording Script")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"EEG output: {eeg_file}")
    if not args.nocamera:
        print(f"Video output: {video_file}")
    else:
        print(f"Video output: DISABLED (--nocamera mode)")
    print(f"Duration: {args.duration}s" if args.duration else "Duration: Until interrupted")
    print()
    
    # Handle device selection and streaming
    muselsl_process = None
    selected_device = None
    stream_name = args.stream_name
    camera_index = args.camera
    
    if args.auto_scan:
        # Scan and select device
        selected_device = select_muse_device()
        if selected_device is None:
            print("No device selected. Exiting.")
            return 1
        
        # Start muselsl stream
        muselsl_process = start_muselsl_stream(
            selected_device['name'],
            include_acc=args.include_acc,
            include_gyro=args.include_gyro
        )
        
        if muselsl_process is None:
            print("Failed to start muselsl stream. Exiting.")
            return 1
        
        # Use the device name as stream name
        stream_name = selected_device['name']
        
        # Wait for LSL stream to be ready
        if not wait_for_lsl_stream(stream_name, timeout=30):
            print("LSL stream not ready. Stopping muselsl...")
            if muselsl_process:
                muselsl_process.terminate()
            return 1
        
        # After successful Muse connection, auto-select camera
        print("\n" + "="*60)
        print("Muse device connected successfully!")
        print("="*60)
        if not args.nocamera:
            camera_index = select_camera(auto_select=True)  # Auto-select non-default camera
        else:
            print("Camera disabled (--nocamera mode)")
    else:
        # Manual mode - assume stream is already running
        print(f"Using stream name: {stream_name}")
        print("(Assuming muselsl is already running in another terminal)")
        # Auto-select camera if using default
        if not args.nocamera and args.camera == 0:  # Only auto-select if using default
            camera_index = select_camera(auto_select=True)
    
    # Create keypress logger
    keypress_logger = KeypressLogger(debug=args.debug) if HAS_PYNPUT else None
    
    # Create recorders
    base_filename = f"eeg_{timestamp}"
    eeg_recorder = EEGRecorderWithReconnect(recording_folder, base_filename, stream_name=stream_name, keypress_logger=keypress_logger, debug=args.debug)
    
    if not args.nocamera:
        video_recorder = VideoRecorder(
            video_file,
            camera_index=camera_index,
            fps=args.fps,
            resolution=resolution
        )
    else:
        video_recorder = None
    
    # Record
    try:
        if args.nocamera:
            # EEG-only recording mode
            record_eeg_only(
                eeg_recorder,
                duration=args.duration,
                keypress_logger=keypress_logger,
                muselsl_process=muselsl_process
            )
        else:
            record_synchronized(
                eeg_recorder,
                video_recorder,
                duration=args.duration,
                show_video=not args.no_preview,
                keypress_logger=keypress_logger
            )
        
        
        # Files saved message for video mode (EEG files are listed by record_eeg_only)
        if not args.nocamera:
            print(f"\nFiles saved:")
            print(f"  Video: {video_file}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up muselsl process if we started it
        if muselsl_process:
            print("\nStopping muselsl stream...")
            muselsl_process.terminate()
            try:
                muselsl_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                muselsl_process.kill()
            print("muselsl stopped.")


if __name__ == "__main__":
    exit(main())


