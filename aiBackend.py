from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
from collections import defaultdict
import math
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

UPLOAD_FOLDER = 'uploads'
IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
DATA_FILE = os.path.join(UPLOAD_FOLDER, 'detections.json')

os.makedirs(IMAGES_FOLDER, exist_ok=True)

detections = []

#load existing detections
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, 'r') as f:
            detections = json.load(f)
        print(f"Loaded {len(detections)} existing detections")
    except:
        print("Starting with empty detections")

def save_detections():
    with open(DATA_FILE, 'w') as f:
        json.dump(detections, f, indent=2)
def analyze_img_w_gemini(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        prompt = """
        Analyze this image and identify any trash or waste items visible.

        For each item you detect, provide:
        1. The type of trash (e.g., plasitic bottle, paper cup, cardboard box, glass bottole, aluminium can, food waste, cigarette butt, etc.)
        2. The material it's made form (plastic, paper, carboard, glass, metal, biodegradable/organic)
        3. You confidence level (0.0 to 1.0)
        4. Disposal category: "recyclable", "compost", or "landfill"

        IMPORTANT: Only identify actual trash/waste items. Ignore people, buildings, vehicles, plants, etc.
        Format your response as a JSON array like this:
        [
            {
                "type": "plastic water bottle",
                "material": "plastic",
                "confidence": 0.95,
                "disposal_category": "recyclable"
            }
        ]
        If no trash is visible, return an empty array: []
        """
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([prompt, image])

        text = response.text.strip()

        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        detected_items = json.loads(text.strip())
        
        print(f"Gemini detected {len(detected_items)} items")
        for item in detected_items:
            print(f"  - {item['type']} ({item['confidence']:.0%} confidence)")
        
        return detected_items
        
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return []

def calculate_heatmap_data():
    heatmap_points=[]
    grid_size = 0.001
    grid = defaultdict(lambda: {'count': 0, 'items': 0})

    for detection in detections:
        gps = detection.get('gps', {})
        lat = gps.get('latitude')
        lon = gps.get('longitude')
        items_count = len(detection.get('detections', []))

        if lat and lon:
            grid_lat = round(lat/grid_size)*grid_size
            grid_lon = round(lon/grid_size)*grid_size
            grid_key = (grid_lat, grid_lon)

            grid[grid_key]['count'] += 1
            grid[grid_key]['items'] += items_count

        for (lat, lon), data in grid.items():
            intensity = min(data['items'] / 10.0, 1.0)
            heatmap_points.append([lat, lon, intensity])
        
        return heatmap_points

def get_ai_insights():
    if not detections:
        return {
            "summary": "No data available yet. Waiting for Wall-e to start detecting trash",
            "recommendations": [],
            "hotspots": []
        }
    total_items = sum(len(d.get('detections', [])) for d in detections)
    trash_types = defaultdict(int)
    for detection in detections:
        for item in detection.get('detections', []):
            trash_types[item.get('type')] += 1
    
    location_data = defaultdict(lambda: {'count': 0, 'types': []})
    for detection in detections:
        gps = detection.get('gps', {})
        location_name = gps.get('location_name', 'Unknown')
        items = detection.get('detections', [])

        location_data[location_name]['count'] += len(items)
        for item in items:
            location_data[location_name]['types'].append(item.get('type'))
    hotspots = sorted(location_data.items(), key=lambda x: x[1]['count'], reverse=True)[:3]

    prompt = f"""
    Analyze this trash detection data and provide environmental insights:

    Total trash items detected: {total_items}
    Number of locations scanned: {len(detections)}

    Trash breakdown:
    {json.dumps(dict(trash_types), indent=2)}

    Top 3 trash hotspots:
    {json.dumps([{loc: data} for loc, data in hotspots], indent=2)}

    Please provide:
    1. A breif summary of the trash situation (2-3 sentences)
    2. 3-5 specific, actionable recommendations for cleanup and prevention
    3. Analysis of which locations need immediate attention and why

    Format your response as JSON with keys: "summary", "recommendations" (array), "hotspot_analysis" (array)
    Be encouraging and solution-focused. Focus on environmental impact and community action
    """

    try:
        #call Gemini API
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        #parse AI response
        text = response.text
        
        #extract JSON from response (Gemini sometimes adds markdown)
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        ai_data = json.loads(text.strip())
        
        #add hotspot details
        ai_data['hotspots'] = [
            {
                'location': loc,
                'items_count': data['count'],
                'top_types': list(set(data['types']))[:3]
            }
            for loc, data in hotspots
        ]
        
        return ai_data
        
    except Exception as e:
        print(f"AI analysis error: {e}")
        #fallback to basic analysis
        return {
            "summary": f"Detected {total_items} items across {len(detections)} locations. Most common: {max(trash_types.items(), key=lambda x: x[1])[0] if trash_types else 'none'}.",
            "recommendations": [
                "Focus cleanup efforts on high-traffic areas",
                "Install more recycling bins in detected hotspots",
                "Organize community cleanup events",
                "Increase public awareness about proper waste disposal"
            ],
            "hotspots": [
                {
                    'location': loc,
                    'items_count': data['count'],
                    'top_types': list(set(data['types']))[:3]
                }
                for loc, data in hotspots
            ]
        }
    
@app.route('/')
def home():
    return f"""
    <html>
    <head>
        <title>Wall-e AI API</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 50px;
                text-align: center;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.95);
                color: #333;
                padding: 40px;
                border-radius: 20px;
                max-width: 600px;
                margin: 0 auto;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            }}
            h1 {{ color: #667eea; }}
            .stat {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 20px 0;
            }}
            .badge {{
                background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
                color: white;
                padding: 5px 15px;
                border-radius: 15x;
                font-size: 0.9em;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Wall-e AI API</h1>
            <span class="badge">Gemini Powered</span>
            <p>Status: <strong style="color: #4caf50;">Running</strong></p>
            <div class="stat">{len(detections)}</div>
            <p> Total Detections Stored</p>

            <h2>New AI Endpoints:</h2>
            <p style="text-align: left; margin: 20px;">
                <strong>POST /upload</strong> - Upload image for AI analysis<br>
                <strong>GET /detections</strong> - Get all detections<br>
                <strong>GET /heatmap</strong> - Get heatmap data<br>
                <strong>GET /ai-insights</strong> - AI-powered analysis<br>
                <strong>GET /stats</strong> - Get statistics<br>
                <strong>DELETE /clear</strong> - Clear all data
            </p>
        </div>
    </body>
    </html>
"""
@app.route('/upload', methods=['POST'])
def upload_detection():
    """Upload endpoint"""
    try:
        image = request.files.get('image')
        metadata_str = request.form.get('metadata')
        
        if not image:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        if not metadata_str:
            return jsonify({"status": "error", "message": "No metadata provided"}), 400
        
        metadata = json.loads(metadata_str)
        detection_id = metadata.get('detection_id', f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"\n{'='*60}")
        print(f"NEW DETECTION: {detection_id}")
        print(f"{'='*60}")
        
        image_bytes = image.read()
        image_filename = f"{detection_id}.jpeg"
        image_path = os.path.join(IMAGES_FOLDER, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        print(f"Saved Image: {image_filename}")
        detected_items = analyze_img_w_gemini(image_bytes)
        if not detected_items:
            print(f"No trash detected")
            return jsonify({
                "status": "success",
                "detection_id": detection_id,
                "message": f"No trash detected",
                "items_found": 0
            }), 200
        detection_record = {
            "detection_id": detection_id,
            "timestamp": metadata.get('timestamp', datetime.now().isoformat()),
            "gps": metadata.get('gps', {}),
            "detections": detected_items,
            "image_url": f"/images/{image_filename}",
            "metadata": {
                "device_id": metadata.get('device_id', 'unknown'),
                "model_version": "gemini-vision",
                "total_items": len(detected_items)
            }
        }

        detections.append(detection_record)
        
        # Keep only last 100 detections
        if len(detections) > 100:
            detections.pop(0)
        
        save_detections()
        
        gps = detection_record.get('gps', {})
        print(f"✓ Detection saved!")
        print(f"  Items: {len(detected_items)}")
        print(f"  GPS: {gps.get('latitude', 'N/A')}, {gps.get('longitude', 'N/A')}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "detection_id": detection_id,
            "message": f"Detected {len(detected_items)} items",
            "items_found": len(detected_items),
            "detections": detected_items
        }), 200
    
    except Exception as e:
        print(f"✗ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/detections', methods=['GET'])
def get_detections():
    limit = request.args.get('limit', type=int)
    result = detections.copy()
    result.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    if limit:
        result = result[:limit]
    
    return jsonify(result), 200

@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    heatmap_data = calculate_heatmap_data()
    
    return jsonify({
        "status": "success",
        "heatmap_points": heatmap_data,
        "total_points": len(heatmap_data)
    }), 200


@app.route('/ai-insights', methods=['GET'])
def ai_insights():
    print("Generating AI insights...")
    insights = get_ai_insights()
    
    return jsonify({
        "status": "success",
        "insights": insights,
        "generated_at": datetime.now().isoformat()
    }), 200

@app.route('/predictions', methods=['GET'])
def get_predictions():
    if len(detections) < 5:
        return jsonify({
            "status": "success",
            "predictions": [],
            "message": "Need more data for predictions"
        }), 200
    
    #analyze time patterns
    hour_data = defaultdict(int)
    location_trends = defaultdict(lambda: {'increasing': False, 'count': 0})
    
    for detection in detections:
        timestamp = datetime.fromisoformat(detection.get('timestamp', ''))
        hour = timestamp.hour
        hour_data[hour] += len(detection.get('detections', []))
        
        gps = detection.get('gps', {})
        location = gps.get('location_name', 'Unknown')
        location_trends[location]['count'] += 1
    
    #find peak hours
    peak_hours = sorted(hour_data.items(), key=lambda x: x[1], reverse=True)[:3]
    
    predictions = {
        "peak_trash_hours": [
            {"hour": hour, "expected_items": count}
            for hour, count in peak_hours
        ],
        "high_risk_locations": [
            {"location": loc, "trend": "increasing" if data['count'] > 2 else "stable"}
            for loc, data in list(location_trends.items())[:5]
        ],
        "recommendation": f"Focus patrols around {peak_hours[0][0]}:00 when trash accumulation is highest."
    }
    
    return jsonify({
        "status": "success",
        "predictions": predictions
    }), 200

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve detection images"""
    try:
        return send_from_directory(IMAGES_FOLDER, filename)
    except:
        return jsonify({"status": "error", "message": "Image not found"}), 404


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics"""
    total_items = sum(len(d.get('detections', [])) for d in detections)
    
    trash_types = {}
    for detection in detections:
        for item in detection.get('detections', []):
            trash_type = item.get('type')
            trash_types[trash_type] = trash_types.get(trash_type, 0) + 1
    
    disposal_categories = {}
    for detection in detections:
        for item in detection.get('detections', []):
            category = item.get('disposal_category', 'unknown')
            disposal_categories[category] = disposal_categories.get(category, 0) + 1
    
    return jsonify({
        "total_detections": len(detections),
        "total_items": total_items,
        "trash_types": trash_types,
        "disposal_categories": disposal_categories
    }), 200


@app.route('/clear', methods=['DELETE'])
def clear_detections():
    """Clear all detections"""
    global detections
    count = len(detections)
    detections = []
    save_detections()
    
    print(f"Cleared {count} detections")
    
    return jsonify({
        "status": "success",
        "message": f"Cleared {count} detections"
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING WALL-E AI API SERVER")
    print("="*60)
    print(f"AI Model: Gemini Vision (gemini-2.0-flash-exp)")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Data file: {os.path.abspath(DATA_FILE)}")
    print(f"Loaded detections: {len(detections)}")
    
    print("="*60)
    print("\nServer running at: http://localhost:5001")
    print("API docs: http://localhost:5001/")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
