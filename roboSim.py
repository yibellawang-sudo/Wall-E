import time
import random
import requests
import json
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

class RobotSimulator:
    def __init__(self, api_endpoint: str = "http://localhost:5001/upload"):
        self.api_endpoint = api_endpoint
        
        self.patrol_route = [
            (43.6532, -79.3832, "University Ave & College St"),
            (43.6555, -79.3845, "Queen's Park"),
            (43.6540, -79.3810, "Bay St & Dundas St"),
            (43.6520, -79.3850, "Spadina Ave & College St"),
            (43.6575, -79.3800, "Yonge St & Bloor St"),
            (43.6510, -79.3790, "King St & Yonge St"),
            (43.6485, -79.3820, "Front St & Union Station"),
            (43.6590, -79.3850, "Harbord St & Spadina"),
        ]

        self.stats = {
            'locations_visited': 0,
            'images_captured': 0,
            'uploads_success': 0,
            'uploads_failed': 0,
            'trash_detected':0
        }
        
        print("Robot Simulator Initialized")
        print(f"Patrol route: {len(self.patrol_route)} locations")
        print(f"API endpoint: {self.api_endpoint}")
        print(f"Images will be analyzed by Gemini Vision\n")

    def generate_realistic_trash_scene(self, num_trash_items: int = None) -> bytes:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(480):
            for j in range(640):   
                base = random.randint(80, 120)
                img[i, j] = [base, base-5, base-10]

        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cv2.rectangle(img, (0, 0), (200, 480), (60, 60, 60), -1)
        img = cv2.addWeighted(img, 0.7, img, 0.3, 0)

        if num_trash_items is None:
            if random.random() > 0.6:
                return self._encode_image(img)
            num_trash_items = random.randint(1, 4)

        trash_visuals = {
            'bottle': self._draw_bottle,
            'can': self._draw_can,
            'bag': self._draw_plastic_bag,
            'paper': self._draw_paper,
            'cup': self._draw_cup,
            'wrapper': self._draw_wrapper
        }
        
        for i in range(num_trash_items):
            trash_type = random.choice(list(trash_visuals.keys()))
            x = random.randint(100, 500)
            y = random.randint(150, 400)
            
            trash_visuals[trash_type](img, x, y)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"RobotCam - {timestamp}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return self._encode_image(img)
        
    def _draw_bottle(self, img, x, y):
        cv2.rectangle(img, (x, y), (x+40, y+80), (180, 140, 100), -1)
        cv2.rectangle(img, (x, y), (x+40, y+80), (140, 100, 60), 2)
        cv2.rectangle(img, (x+10, y-10), (x+30, y), (200, 200, 200), -1)
        cv2.rectangle(img, (x+5, y+20), (x+35, y+40), (255, 255, 255), -1)
    
    def _draw_can(self, img, x, y):
        cv2.ellipse(img, (x+20, y), (20, 8), 0, 0, 360, (200, 200, 210), -1)
        cv2.rectangle(img, (x, y), (x+40, y+60), (210, 210, 220), -1)
        cv2.ellipse(img, (x+20, y+60), (20, 8), 0, 0, 360, (180, 180, 190), -1)
        cv2.line(img, (x+10, y+10), (x+10, y+50), (240, 240, 250), 2)
    
    def _draw_plastic_bag(self, img, x, y):
        pts = np.array([
            [x, y], [x+30, y-10], [x+60, y+5], 
            [x+50, y+40], [x+20, y+45], [x+5, y+30]
        ], np.int32)
        cv2.fillPoly(img, [pts], (240, 240, 250))
        cv2.polylines(img, [pts], True, (200, 200, 210), 2)
    
    def _draw_paper(self, img, x, y):
        cv2.rectangle(img, (x, y), (x+50, y+40), (245, 245, 250), -1)
        cv2.rectangle(img, (x, y), (x+50, y+40), (200, 200, 200), 2)
        for i in range(3):
            y_line = y + 10 + i*10
            cv2.line(img, (x+5, y_line), (x+45, y_line), (220, 220, 220), 1)
    
    def _draw_cup(self, img, x, y):
        pts = np.array([[x+10, y], [x+5, y+50], [x+45, y+50], [x+40, y]], np.int32)
        cv2.fillPoly(img, [pts], (255, 250, 240))
        cv2.polylines(img, [pts], True, (200, 180, 160), 2)
        cv2.ellipse(img, (x+25, y), (18, 6), 0, 0, 360, (220, 200, 180), -1)
    
    def _draw_wrapper(self, img, x, y):
        pts = np.array([
            [x, y+10], [x+15, y], [x+40, y+5], 
            [x+35, y+20], [x+10, y+25]
        ], np.int32)
        colors = [(255, 200, 100), (200, 100, 255), (100, 255, 200)]
        color = random.choice(colors)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, tuple(c-40 for c in color), 1)
    def _encode_image(self, img) -> bytes:
        _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes()
    def upload_detection(self, gps_data: tuple, image_bytes: bytes):
        lat, lon, location_name = gps_data
        detection_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        metadata = {
            "detection_id": detection_id,
            "timestamp": datetime.now().isoformat(),
            "device_id": "robot_simulator_001",
            "gps": {
                "latitude": lat,
                "longitude": lon,
                "accuracy": 5.0,
                "location_name": location_name
            }
        }

        try:
            response = requests.post(
                self.api_endpoint,
                files={'image': ('detection.jpg', image_bytes, 'image/jpeg')},
                data={'metadata': json.dumps(metadata)},
                timeout = 30
            )
            if response.status_code == 200:
                result = response.json()
                self.stats['uploads_success'] += 1

                items_found = result.get('items_found', 0)
                if items_found > 0:
                    self.stats['trash_detected'] += items_found
                    print(f"Upload sucessful: {items_found} items detected by Gemini")

                    detections = result.get('detections', [])
                    for item in detections[:3]:
                        print(f" - {item['type']} ({item['confidence']:.0%})")
                else:
                    print(f"Upload sucessful, no trash detected")
                return True
            else:
                self.stats['uploads_failed'] += 1
                print(f"Upload failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.stats['uploads_failed'] += 1
            print(f"Upload error: {e}")
            return False
            
    def patrol(self, loops: int = 1, travel_time: float = 3.0, 
               detection_delay: float = 2.0):
        """
        Simulate robot patrol
        
        Args:
            loops: Number of times to complete the route
            travel_time: Seconds between locations (simulates robot movement)
            detection_delay: Seconds to "analyze" each location
        """
        print("="*60)
        print("STARTING")
        print("="*60)
        print(f"Route loops: {loops}")
        print(f"Travel time: {travel_time}s between locations")
        print(f"Detection delay: {detection_delay}s per location")
        print("="*60 + "\n")
        
        try:
            for loop in range(loops):
                print(f"\nLoop {loop + 1}/{loops}")
                print("-" * 60)
                
                for i, location in enumerate(self.patrol_route):
                    lat, lon, name = location
                    self.stats['locations_visited'] += 1
                    
                    print(f"\nLocation {i+1}/{len(self.patrol_route)}: {name}")
                    print(f"GPS: {lat:.4f}, {lon:.4f}")
                    
                    # Simulate travel time
                    if i > 0:
                        print(f"Traveling... ({travel_time}s)")
                        time.sleep(travel_time)
                    
                    # Simulate img capture
                    print(f"Capturing image...({detection_delay}s)")
                    time.sleep(detection_delay)
                    image_bytes = self.generate_realistic_trash_scene()
                    self.stats['images_captured'] += 1

                    print(f"Uploading to AI for analysis...")
                    self.upload_detection(location, image_bytes)
                
                if loop < loops - 1:
                    print(f"\nCompleting loop {loop + 1}, starting next loop...\n")
                    time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n\nPatrol interrupted by user")
        
        finally:
            self.print_summary()
    
    def continuous_patrol(self, detection_interval: float = 5.0):
        print("="*60)
        print("STARTING CONTINUOUS PATROL")
        print("="*60)
        print(f"Detection interval: {detection_interval}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Pick random location from route
                location = random.choice(self.patrol_route)
                lat, lon, name = location
                
                # Add small random offset (simulates robot movement)
                lat += random.uniform(-0.0005, 0.0005)
                lon += random.uniform(-0.0005, 0.0005)
                
                self.stats['locations_visited'] += 1
                
                print(f"\n{name}")
                print(f"GPS: {lat:.4f}, {lon:.4f}")
                
                # Capture and upload image
                print(f"Capturing image...")
                image_bytes = self.generate_realistic_trash_scene()
                self.stats['images_captured'] += 1
                
                print(f"Uploading for AI analysis...")
                self.upload_detection((lat, lon, name), image_bytes)
                
                time.sleep(detection_interval)
        except KeyboardInterrupt:
            print("\n\nPatrol stopped")
            self.print_summary()
    
    def print_summary(self):
        """Print patrol statistics"""
        print("\n" + "="*60)
        print("PATROL SUMMARY")
        print("="*60)
        print(f"Locations visited:  {self.stats['locations_visited']}")
        print(f"Images Captured:  {self.stats['images_captured']}")
        print(f"Trash items found:  {self.stats['trash_detected']}")
        print(f"Successful uploads: {self.stats['uploads_success']}")
        print(f"Failed uploads:     {self.stats['uploads_failed']}")
        
        if self.stats['uploads_success'] + self.stats['uploads_failed'] > 0:
            success_rate = (self.stats['uploads_success'] / 
                          (self.stats['uploads_success'] + self.stats['uploads_failed'])) * 100
            print(f"Upload success rate: {success_rate:.1f}%")
        if self.stats['images_captured'] > 0 and self.stats['trash_detected'] > 0:
            avg_items = self.stats['trash_detected'] / self.stats['images_captured']
            print(f"Avg items per image: {avg_items:.1f}")
        print("="*60 + "\n")


def main():
    print("\nTrashBot Simulator")
    print("="*60)
    print("Choose simulation mode:")
    print("1. Quick Demo (1 loop, fast)")
    print("2. Full Patrol (3 loops, realistic)")
    print("3. Continuous (runs until stopped)")
    print("4. Custom")
    print("="*60)
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    # Initialize
    api_endpoint = input("API endpoint (press Enter for http://localhost:5001/upload): ").strip()
    if not api_endpoint:
        api_endpoint = "http://localhost:5001/upload"
    
    simulator = RobotSimulator(api_endpoint=api_endpoint)
    
    print("\n")
    
    if choice == "1":
        # Quick demo
        simulator.patrol(loops=1, travel_time=1.0, detection_delay=0.5)
    
    elif choice == "2":
        # Full patrol
        simulator.patrol(loops=3, travel_time=3.0, detection_delay=2.0)
    
    elif choice == "3":
        # Continuous
        interval = input("Detection interval in seconds (default 5): ").strip()
        interval = float(interval) if interval else 5.0
        simulator.continuous_patrol(detection_interval=interval)
    
    elif choice == "4":
        # Custom
        loops = int(input("Number of loops: "))
        travel_time = float(input("Travel time between locations (seconds): "))
        detection_delay = float(input("Detection delay (seconds): "))
        simulator.patrol(loops=loops, travel_time=travel_time, 
                        detection_delay=detection_delay)
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
