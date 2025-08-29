import cv2
import os
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict

MQTT_BROKER = os.environ.get('MQTT_BROKER', "10.249.70.108")
MQTT_PORT = 1883
MQTT_USERNAME = os.environ.get('MQTT_USERNAME', "cpsmagang")
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', "cpsjaya123")
DEVICE_IP_ADDRESS = os.environ.get('DEVICE_IP_ADDRESS', "10.249.70.108")

STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

# Enhanced motion configuration for crowded rooms
MOTION_CONFIG = {
    "enabled": True,
    "detection_duration": 1.5,
    "movement_threshold": 45.0,
    "position_buffer_size": 20,
    "confidence_threshold": 0.6,
    "stable_detection_frames": 8,
    "motion_cooldown": 2.0,
    "min_movement_points": 3,
    "relative_movement_threshold": 0.08,
    "keypoint_stability_threshold": 0.03,
    "min_stable_keypoints": 6,
    "max_tracked_persons": 10,
    "person_timeout": 5.0,
    "significant_motion_threshold": 0.12,
    "crowd_motion_weight": 0.7,
    "individual_motion_weight": 0.3,
    "area_division_grid": (3, 3),
    "min_area_coverage": 0.15
}

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

    # Enhanced multi-person motion tracker
    motion_tracker = {
        "persons": {},  # Track multiple persons
        "global_motion_history": deque(maxlen=30),
        "area_motion_grid": defaultdict(lambda: deque(maxlen=15)),
        "last_cleanup": time.time(),
        "room_occupancy_level": 0,
        "significant_motion_detected": False,
        "motion_start_time": None,
        "last_significant_motion": None,
        "motion_triggered": False,
        "frame_motion_scores": deque(maxlen=10)
    }

    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            exit()
    else:
        exit()

except Exception as e:
    exit()

consecutive_detections = 0
fps_buffer = []
fps_avg_len = 30

def calculate_person_id(center_point, existing_persons, max_distance=80):
    """Assign person ID based on proximity to existing tracked persons"""
    if not existing_persons:
        return f"person_{len(existing_persons)}"
    
    min_distance = float('inf')
    closest_id = None
    
    for person_id, person_data in existing_persons.items():
        if person_data["positions"]:
            last_pos = person_data["positions"][-1]
            distance = np.sqrt((center_point[0] - last_pos[0])**2 + (center_point[1] - last_pos[1])**2)
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_id = person_id
    
    return closest_id if closest_id else f"person_{len(existing_persons)}"

def get_stable_keypoints_enhanced(keypoints):
    """Enhanced keypoint extraction with noise filtering"""
    if keypoints is None or len(keypoints) == 0:
        return None
    
    all_stable_keypoints = []
    
    for person_keypoints in keypoints:
        if len(person_keypoints) == 0:
            continue
            
        stable_keypoints = []
        for i in range(len(person_keypoints)):
            if len(person_keypoints[i]) >= 3 and person_keypoints[i][2] > MOTION_CONFIG["confidence_threshold"]:
                # Apply slight smoothing to reduce noise
                x, y, conf = person_keypoints[i]
                stable_keypoints.append([float(x), float(y), float(conf)])
        
        if len(stable_keypoints) >= MOTION_CONFIG["min_stable_keypoints"]:
            all_stable_keypoints.append(np.array(stable_keypoints))
    
    return all_stable_keypoints if all_stable_keypoints else None

def calculate_pose_center_enhanced(stable_keypoints):
    """Calculate pose center with weighted importance for key body parts"""
    if stable_keypoints is None or len(stable_keypoints) == 0:
        return None
    
    # Weight important keypoints more (torso, shoulders, hips)
    important_indices = [5, 6, 11, 12]  # shoulders and hips
    
    if len(stable_keypoints) > max(important_indices):
        important_points = stable_keypoints[important_indices]
        weights = important_points[:, 2]  # confidence scores as weights
        
        if np.sum(weights) > 0:
            center_x = np.average(important_points[:, 0], weights=weights)
            center_y = np.average(important_points[:, 1], weights=weights)
        else:
            center_x = np.mean(stable_keypoints[:, 0])
            center_y = np.mean(stable_keypoints[:, 1])
    else:
        center_x = np.mean(stable_keypoints[:, 0])
        center_y = np.mean(stable_keypoints[:, 1])
    
    return (center_x, center_y)

def calculate_motion_significance(current_keypoints, reference_keypoints):
    """Calculate motion significance with focus on meaningful body movements"""
    if current_keypoints is None or reference_keypoints is None:
        return 0.0
    
    if len(current_keypoints) != len(reference_keypoints):
        return 0.0
    
    # Focus on key body parts for motion detection
    key_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # head, arms, torso, legs
    
    significant_movements = 0
    total_movement = 0.0
    
    for i in key_indices:
        if i < len(current_keypoints) and i < len(reference_keypoints):
            curr_point = current_keypoints[i][:2]
            ref_point = reference_keypoints[i][:2]
            
            movement = np.sqrt(np.sum((curr_point - ref_point) ** 2))
            total_movement += movement
            
            # Check if this is significant movement (not just noise)
            if movement > MOTION_CONFIG["movement_threshold"] * 0.3:
                significant_movements += 1
    
    # Normalize by image size and number of keypoints checked
    normalized_movement = total_movement / (len(key_indices) * np.sqrt(resW**2 + resH**2))
    significance_score = (significant_movements / len(key_indices)) * normalized_movement
    
    return min(significance_score * 10, 1.0)  # Scale to 0-1 range

def analyze_crowd_motion():
    """Analyze overall crowd motion patterns"""
    current_time = time.time()
    
    # Clean up old person data
    if current_time - motion_tracker["last_cleanup"] > 2.0:
        persons_to_remove = []
        for person_id, person_data in motion_tracker["persons"].items():
            if current_time - person_data["last_seen"] > MOTION_CONFIG["person_timeout"]:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del motion_tracker["persons"][person_id]
        
        motion_tracker["last_cleanup"] = current_time
    
    # Calculate crowd motion metrics
    active_persons = 0
    total_motion_score = 0.0
    area_coverage = set()
    
    grid_w, grid_h = MOTION_CONFIG["area_division_grid"]
    
    for person_id, person_data in motion_tracker["persons"].items():
        if current_time - person_data["last_seen"] < 1.0:  # Recently seen
            active_persons += 1
            total_motion_score += person_data["current_motion_score"]
            
            # Calculate area coverage
            if person_data["positions"]:
                pos = person_data["positions"][-1]
                grid_x = int((pos[0] / resW) * grid_w)
                grid_y = int((pos[1] / resH) * grid_h)
                area_coverage.add((grid_x, grid_y))
    
    motion_tracker["room_occupancy_level"] = active_persons
    
    # Determine if there's significant motion
    avg_motion_score = total_motion_score / max(active_persons, 1)
    area_coverage_ratio = len(area_coverage) / (grid_w * grid_h)
    
    # Adaptive thresholding based on crowd size
    motion_threshold = MOTION_CONFIG["significant_motion_threshold"]
    if active_persons > 5:  # Crowded room
        motion_threshold *= 1.5  # Be less sensitive in crowded rooms
    elif active_persons < 2:  # Few people
        motion_threshold *= 0.7  # Be more sensitive with few people
    
    significant_motion = (avg_motion_score > motion_threshold or 
                         (area_coverage_ratio > MOTION_CONFIG["min_area_coverage"] and avg_motion_score > motion_threshold * 0.5))
    
    motion_tracker["frame_motion_scores"].append(avg_motion_score)
    
    # Smooth motion detection over multiple frames
    if len(motion_tracker["frame_motion_scores"]) >= 5:
        recent_avg = np.mean(list(motion_tracker["frame_motion_scores"])[-5:])
        significant_motion = recent_avg > motion_threshold
    
    return significant_motion, active_persons

def update_multi_person_motion_detection(keypoints):
    """Enhanced motion detection for multiple persons"""
    current_time = time.time()
    all_stable_keypoints = get_stable_keypoints_enhanced(keypoints)
    
    if all_stable_keypoints:
        # Process each detected person
        for person_keypoints in all_stable_keypoints:
            center_point = calculate_pose_center_enhanced(person_keypoints)
            
            if center_point is None:
                continue
            
            person_id = calculate_person_id(center_point, motion_tracker["persons"])
            
            # Initialize or update person data
            if person_id not in motion_tracker["persons"]:
                motion_tracker["persons"][person_id] = {
                    "positions": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
                    "timestamps": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
                    "keypoint_history": deque(maxlen=10),
                    "last_seen": current_time,
                    "current_motion_score": 0.0,
                    "stable_frames": 0
                }
            
            person_data = motion_tracker["persons"][person_id]
            person_data["last_seen"] = current_time
            person_data["positions"].append(center_point)
            person_data["timestamps"].append(current_time)
            person_data["keypoint_history"].append(person_keypoints)
            
            # Calculate motion score for this person
            if len(person_data["keypoint_history"]) >= 2:
                recent_kp = person_data["keypoint_history"][-1]
                prev_kp = person_data["keypoint_history"][-2]
                motion_score = calculate_motion_significance(recent_kp, prev_kp)
                person_data["current_motion_score"] = motion_score
                
                # Update stability counter
                if motion_score < MOTION_CONFIG["keypoint_stability_threshold"]:
                    person_data["stable_frames"] = min(person_data["stable_frames"] + 1, 10)
                else:
                    person_data["stable_frames"] = 0
    
    # Analyze overall crowd motion
    significant_motion, active_persons = analyze_crowd_motion()
    
    # Update global motion state
    if significant_motion:
        if not motion_tracker["significant_motion_detected"]:
            motion_tracker["motion_start_time"] = current_time
            motion_tracker["significant_motion_detected"] = True
        motion_tracker["last_significant_motion"] = current_time
        
        # Check if motion duration threshold is met
        if (current_time - motion_tracker["motion_start_time"]) >= MOTION_CONFIG["detection_duration"]:
            motion_tracker["motion_triggered"] = True
    else:
        # Check for motion cooldown
        if (motion_tracker["last_significant_motion"] and 
            current_time - motion_tracker["last_significant_motion"] > MOTION_CONFIG["motion_cooldown"]):
            motion_tracker["significant_motion_detected"] = False
            motion_tracker["motion_triggered"] = False
            motion_tracker["motion_start_time"] = None
    
    return active_persons > 0

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload)

def on_message(client, userdata, msg):
    global devices
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
        
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]

    except Exception as e:
        pass

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            break

        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Enhanced person detection with multiple person support
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0
        keypoints = results[0].keypoints.data.cpu().numpy() if pose_found else None
        
        persons_detected = update_multi_person_motion_detection(keypoints)

        if persons_detected:
            consecutive_detections = min(consecutive_detections + 1, 20)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        # Enhanced logic for device activation
        if MOTION_CONFIG["enabled"]:
            # Consider both person presence and motion significance
            should_be_active = (consecutive_detections >= MOTION_CONFIG["stable_detection_frames"] and 
                              motion_tracker["motion_triggered"] and
                              motion_tracker["room_occupancy_level"] > 0)
        else:
            should_be_active = consecutive_detections >= MOTION_CONFIG["stable_detection_frames"]
        
        should_be_inactive = (consecutive_detections <= 2 or 
                            motion_tracker["room_occupancy_level"] == 0)
        
        now = datetime.now().time()

        # Device control logic
        for name, device in devices.items():
            if device["mode"] == "auto":
                if should_be_active and device["state"] == 0:
                    device["instance"].on()
                    device["state"] = 1
                elif should_be_inactive and device["state"] == 1:
                    device["instance"].off()
                    device["state"] = 0

                # MQTT reporting with enhanced data
                if should_be_active and not device["is_person_reported"]:
                    device["is_person_reported"] = True
                    payload = json.dumps({
                        "device": name, 
                        "motion_detected": True,
                        "person_count": motion_tracker["room_occupancy_level"],
                        "motion_level": "significant" if motion_tracker["motion_triggered"] else "minimal"
                    })
                    client.publish(SENSOR_TOPIC, payload)
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    payload = json.dumps({
                        "device": name, 
                        "motion_cleared": True,
                        "person_count": 0
                    })
                    client.publish(SENSOR_TOPIC, payload)

            elif device["mode"] == "scheduled":
                try:
                    if device["schedule_on"] and device["schedule_off"]:
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

        # Display enhanced information
        y_pos = 30
        for name, device in devices.items():
            mode_text = f"{name.upper()}: {device['mode'].upper()}"
            status_text = f"{'ON' if device['state'] == 1 else 'OFF'}"
            
            color_mode = (0, 255, 255)
            color_status = (0, 255, 0) if device['state'] == 1 else (0, 0, 255)
            
            cv2.putText(annotated_frame, f"{mode_text} - {status_text}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
            y_pos += 25

        # Display crowd information
        occupancy_text = f"People: {motion_tracker['room_occupancy_level']}"
        motion_text = f"Motion: {'ACTIVE' if motion_tracker['motion_triggered'] else 'IDLE'}"
        
        cv2.putText(annotated_frame, occupancy_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(annotated_frame, motion_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if motion_tracker['motion_triggered'] else (128, 128, 128), 2)

        # FPS calculation and display
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.1f}', 
                       (resW - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()
    for device in devices.values():
        device["instance"].close()
    client.loop_stop()
    client.disconnect()