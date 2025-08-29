import cv2
import os
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime
from collections import deque

#config
MQTT_BROKER = os.environ['MQTT_BROKER']
MQTT_PORT = 1883
MQTT_USERNAME = os.environ['MQTT_USERNAME']
MQTT_PASSWORD = os.environ['MQTT_PASSWORD']
DEVICE_IP_ADDRESS = os.environ['DEVICE_IP_ADDRESS']

# TOPIK MQTT yang dipake
STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

# Motion Detection Configuration
MOTION_CONFIG = {
    "enabled": True,
    "detection_duration": 3.0,
    "movement_threshold": 30.0,
    "position_buffer_size": 30,
    "confidence_threshold": 0.5,
    "stable_detection_frames": 8,
    "motion_cooldown": 2.0,
    "min_movement_points": 3,
    "motion_sensitivity": "medium"
}

if MOTION_CONFIG["motion_sensitivity"] == "low":
    MOTION_CONFIG["movement_threshold"] = 50.0
    MOTION_CONFIG["detection_duration"] = 5.0
elif MOTION_CONFIG["motion_sensitivity"] == "high":
    MOTION_CONFIG["movement_threshold"] = 15.0
    MOTION_CONFIG["detection_duration"] = 1.5
else:
    MOTION_CONFIG["movement_threshold"] = 30.0
    MOTION_CONFIG["detection_duration"] = 3.0

# Inisialisasi Perangkat dan Model
try:
    model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose")
    cam_source = "usb0"
    resW, resH = 640, 480

    devices = {
        "lamp": {
            "instance": LED(26),
            "state": 0,  
            "mode": "auto", 
            "schedule_on": None, 
            "schedule_off": None,
            "is_person_reported": False
        },
        "fan": {
            "instance": LED(19),
            "state": 0,
            "mode": "auto",
            "schedule_on": None,
            "schedule_off": None,
            "is_person_reported": False
        }
    }

    motion_tracker = {
        "person_positions": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "position_timestamps": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "is_motion_detected": False,
        "motion_start_time": None,
        "last_motion_time": None,
        "person_detected": False,
        "motion_triggered": False
    }

    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            print("Gagal membuka kamera.")
            exit()
    else:
        print("Tidak ada kamera terdeteksi.")
        exit()
    print("Kamera siap.")

except Exception as e:
    print(f"Error saat inisialisasi: {e}")
    exit()

consecutive_detections = 0
fps_buffer = []
fps_avg_len = 50

def calculate_center_keypoint(keypoints):
    if keypoints is None or len(keypoints) == 0:
        return None
    
    kp = keypoints[0]
    
    valid_points = []
    for i in range(len(kp)):
        if len(kp[i]) >= 3 and kp[i][2] > MOTION_CONFIG["confidence_threshold"]:
            valid_points.append((kp[i][0], kp[i][1]))
    
    if not valid_points:
        return None
        
    center_x = sum([p[0] for p in valid_points]) / len(valid_points)
    center_y = sum([p[1] for p in valid_points]) / len(valid_points)
    
    return (center_x, center_y)

def detect_motion():
    if not MOTION_CONFIG["enabled"] or len(motion_tracker["person_positions"]) < MOTION_CONFIG["min_movement_points"]:
        return False
    
    current_time = time.time()
    
    motion_duration = 0
    total_movement = 0
    movement_count = 0
    
    positions = list(motion_tracker["person_positions"])
    timestamps = list(motion_tracker["position_timestamps"])
    
    for i in range(len(positions) - 1):
        time_diff = timestamps[i+1] - timestamps[i]
        if time_diff > 0 and time_diff <= MOTION_CONFIG["detection_duration"]:
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > MOTION_CONFIG["movement_threshold"]:
                motion_duration += time_diff
                total_movement += distance
                movement_count += 1
    
    return (motion_duration >= MOTION_CONFIG["detection_duration"] and 
            total_movement > MOTION_CONFIG["movement_threshold"] and
            movement_count >= MOTION_CONFIG["min_movement_points"])

def update_motion_detection(keypoints):
    global motion_tracker
    
    current_time = time.time()
    center_point = calculate_center_keypoint(keypoints)
    
    if center_point is not None:
        motion_tracker["person_detected"] = True
        
        motion_tracker["person_positions"].append(center_point)
        motion_tracker["position_timestamps"].append(current_time)
        
        if detect_motion():
            if not motion_tracker["is_motion_detected"]:
                motion_tracker["motion_start_time"] = current_time
                motion_tracker["is_motion_detected"] = True
            
            motion_tracker["last_motion_time"] = current_time
            
            if (current_time - motion_tracker["motion_start_time"]) >= MOTION_CONFIG["detection_duration"]:
                motion_tracker["motion_triggered"] = True
        
    else:
        motion_tracker["person_detected"] = False
        
        if (motion_tracker["last_motion_time"] and 
            current_time - motion_tracker["last_motion_time"] > MOTION_CONFIG["motion_cooldown"]):
            motion_tracker["is_motion_detected"] = False
            motion_tracker["motion_triggered"] = False
            motion_tracker["motion_start_time"] = None

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        print(f"SUBSCRIBE ke topik settings: {SETTINGS_UPDATE_TOPIC}")

        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload)
        print(f"PUBLISH: Mengirim status ONLINE ke {STATUS_TOPIC}")
    else:
        print(f"Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    global devices
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        
        if msg.topic == ACTION_TOPIC:
            device_name = payload.get("device")
            action = payload.get("action")
            
            if device_name in devices and action in ["turn_on", "turn_off"]:
                devices[device_name]["mode"] = "manual"
                if action == "turn_on":
                    devices[device_name]["instance"].on()
                    devices[device_name]["state"] = 1
                elif action == "turn_off":
                    devices[device_name]["instance"].off()
                    devices[device_name]["state"] = 0
                print(f"AKSI MANUAL: '{action}' pada '{device_name}'. Mode diubah ke MANUAL.")
        
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                        print(f"SETTINGS UPDATE: Mode '{device_name}' diubah menjadi {new_mode.upper()}")
                
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                    print(f"SETTINGS UPDATE: Jadwal ON '{device_name}' diatur ke {payload['schedule_on']}")
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]
                    print(f"SETTINGS UPDATE: Jadwal OFF '{device_name}' diatur ke {payload['schedule_off']}")

    except Exception as e:
        print(f"Error memproses pesan: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("\nSistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("Peringatan: Gagal mengambil frame.")
            break

        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        keypoints = results[0].keypoints.data.cpu().numpy() if pose_found else None
        update_motion_detection(keypoints)

        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 10)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        if MOTION_CONFIG["enabled"]:
            should_be_active = (consecutive_detections >= MOTION_CONFIG["stable_detection_frames"] and 
                              motion_tracker["motion_triggered"])
        else:
            should_be_active = consecutive_detections >= MOTION_CONFIG["stable_detection_frames"]
        
        should_be_inactive = consecutive_detections <= 0 or not motion_tracker["person_detected"]
        
        now = datetime.now().time()

        for name, device in devices.items():
            if device["mode"] == "auto":
                if should_be_active and device["state"] == 0:
                    device["instance"].on()
                    device["state"] = 1
                elif should_be_inactive and device["state"] == 1:
                    device["instance"].off()
                    device["state"] = 0

                if should_be_active and not device["is_person_reported"]:
                    device["is_person_reported"] = True
                    payload = json.dumps({"device": name, "motion_detected": True})
                    client.publish(SENSOR_TOPIC, payload)
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    payload = json.dumps({"device": name, "motion_cleared": True})
                    client.publish(SENSOR_TOPIC, payload)

            elif device["mode"] == "scheduled":
                try:
                    on_time = datetime.strptime(device["schedule_on"], "%H:%M").time()
                    off_time = datetime.strptime(device["schedule_off"], "%H:%M").time()
                    
                    is_active_time = False
                    if on_time < off_time:
                        if on_time <= now < off_time:
                            is_active_time = True
                    else:
                        if now >= on_time or now < off_time:
                            is_active_time = True
                    
                    if is_active_time and device["state"] == 0:
                        device["instance"].on()
                        device["state"] = 1
                    elif not is_active_time and device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0
                except (ValueError, TypeError):
                    if device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0
        y_pos = 30
        for name, device in devices.items():
            mode_text = f"{name.upper()} Mode: {device['mode'].upper()}"
            status_text = f"{name.upper()} Status: {'ON' if device['state'] == 1 else 'OFF'}"
            
            color_mode = (0, 255, 255) 
            color_status = (0, 255, 0) if device['state'] == 1 else (0, 0, 255)
            
            cv2.putText(annotated_frame, mode_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_mode, 2)
            y_pos += 30
            cv2.putText(annotated_frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_status, 2)
            y_pos += 40

        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (resW - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("\nMembersihkan sumber daya...")
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    print(f"PUBLISH: Mengirim status OFFLINE ke {STATUS_TOPIC}")
    time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()
    for device in devices.values():
        device["instance"].close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")