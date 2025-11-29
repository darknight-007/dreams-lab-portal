# 🚀 Hackathon Quick Start Guide
## Get Started in 30 Minutes!

**Goal:** Have the AI infrastructure running and ready to develop

---

## ☑️ Pre-Hackathon Checklist (Do This NOW)

### 1. Get API Keys (15 minutes)

```bash
# Create .env.ai file
cd /home/jdas/dreams-lab-website-server/deepgis-xr
nano .env.ai
```

Add these keys:
```bash
# OpenAI (for GPT-4 and Whisper)
OPENAI_API_KEY=sk-...  # Get from https://platform.openai.com/api-keys

# Anthropic (backup LLM)
ANTHROPIC_API_KEY=sk-ant-...  # Get from https://console.anthropic.com/

# ElevenLabs (text-to-speech)
ELEVENLABS_API_KEY=...  # Get from https://elevenlabs.io/

# Optional but recommended
HUGGINGFACE_TOKEN=hf_...  # Get from https://huggingface.co/settings/tokens
```

### 2. Install AI Dependencies (10 minutes)

```bash
cd /home/jdas/dreams-lab-website-server/deepgis-xr

# Create AI requirements file
cat > requirements-ai.txt << 'EOF'
# LLM & NLP
openai==1.3.0
anthropic==0.7.0
langchain==0.1.0
langchain-openai==0.0.2
chromadb==0.4.18
tiktoken==0.5.1

# Computer Vision (lightweight version for quick start)
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
pillow==10.1.0

# API & Services
fastapi==0.105.0
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6

# Utilities
numpy==1.26.2
requests==2.31.0
python-dotenv==1.0.0
EOF

# Install (use existing venv or create new one)
source venv/bin/activate
pip install -r requirements-ai.txt
```

### 3. Create Basic AI Service Structure (5 minutes)

```bash
cd /home/jdas/dreams-lab-website-server/deepgis-xr

# Create directory structure
mkdir -p ai_services/{geogpt,autolabel,forecast,voice,search,shared}
mkdir -p models/embeddings

# Create init files
touch ai_services/__init__.py
touch ai_services/geogpt/__init__.py
touch ai_services/autolabel/__init__.py
touch ai_services/forecast/__init__.py
touch ai_services/voice/__init__.py
touch ai_services/search/__init__.py
touch ai_services/shared/__init__.py
```

### 4. Test OpenAI API (2 minutes)

```bash
# Quick test
python3 << 'EOF'
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env.ai')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[{"role": "user", "content": "Say 'AI Hackathon Ready!' in a creative way"}],
    max_tokens=50
)

print(response.choices[0].message.content)
print("\n✅ OpenAI API Working!")
EOF
```

If this works, you're READY TO GO! 🎉

---

## 🏃 Day 1 Quick Wins (Start Here!)

### Option A: Start with GeoGPT (Easiest)

**Create a minimal chat interface in 30 minutes:**

```bash
cd /home/jdas/dreams-lab-website-server/deepgis-xr/ai_services/geogpt
```

**File: `simple_chat.py`**
```python
#!/usr/bin/env python3
"""
Minimal GeoGPT - Get chatting with DeepGIS in 30 minutes!
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

# Load environment variables
load_dotenv('../../.env.ai')

app = FastAPI(title="GeoGPT Minimal")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# System prompt
SYSTEM_PROMPT = """You are GeoGPT, an AI assistant for the DeepGIS-XR geospatial platform.

Available functions:
- load_layer(layer_id): Load a raster/vector layer
- fly_to(lon, lat, height): Move camera to location
- measure_distance(points): Calculate distance between points
- get_layer_info(layer_id): Get layer metadata

When user asks to do something, explain what you would do, then return a JSON action.

Example:
User: "Show me Mount Everest"
You: "I'll fly the camera to Mount Everest at coordinates 86.9250°E, 27.9881°N"
Action: {"action": "fly_to", "lon": 86.9250, "lat": 27.9881, "height": 50000}
"""

@app.get("/")
async def root():
    return {"status": "GeoGPT is alive!", "version": "0.1.0"}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            user_message = message.get("message", "")
            
            # Add to history
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=conversation_history,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content
            
            # Add to history
            conversation_history.append({
                "role": "assistant",
                "content": ai_message
            })
            
            # Send response
            await websocket.send_json({
                "message": ai_message,
                "status": "success"
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    print("🚀 Starting GeoGPT Minimal...")
    print("📡 WebSocket available at: ws://localhost:9001/ws/chat")
    print("🌐 Test at: http://localhost:9001")
    uvicorn.run(app, host="0.0.0.0", port=9001)
```

**Run it:**
```bash
chmod +x simple_chat.py
python3 simple_chat.py
```

**Test it:**
```bash
# In another terminal
curl http://localhost:9001
# Should return: {"status":"GeoGPT is alive!","version":"0.1.0"}
```

**Connect from browser console:**
```javascript
// Open https://deepgis.org/label/3d/topology/legacy/
// Open browser console (F12)

const ws = new WebSocket('ws://localhost:9001/ws/chat');

ws.onopen = () => {
    console.log('✅ Connected to GeoGPT');
    ws.send(JSON.stringify({message: "Show me Mount Everest"}));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('🤖 GeoGPT:', data.message);
};
```

**YOU JUST BUILT AN AI CHAT INTERFACE! 🎉**

---

### Option B: Start with AutoLabel (More Visual)

**Create a simple segmentation API:**

```bash
cd /home/jdas/dreams-lab-website-server/deepgis-xr/ai_services/autolabel
```

**File: `simple_segment.py`**
```python
#!/usr/bin/env python3
"""
Minimal AutoLabel - Image segmentation in 30 minutes!
Note: We'll use a simple algorithm first, SAM integration comes later
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = FastAPI(title="AutoLabel Minimal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/segment/simple")
async def simple_segment(file: UploadFile = File(...)):
    """
    Simple edge-based segmentation (placeholder for SAM)
    """
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Simple segmentation (Canny edge detection + contours)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Convert mask to base64
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode()
    
    return {
        "status": "success",
        "num_features": len(contours),
        "mask": f"data:image/png;base64,{mask_b64}"
    }

@app.get("/")
async def root():
    return {"status": "AutoLabel is alive!", "note": "Upload an image to /segment/simple"}

if __name__ == "__main__":
    print("🚀 Starting AutoLabel Minimal...")
    print("📡 API available at: http://localhost:9002")
    uvicorn.run(app, host="0.0.0.0", port=9002)
```

**Run it:**
```bash
chmod +x simple_segment.py
python3 simple_segment.py
```

**Test it:**
```bash
# Upload a test image
curl -X POST "http://localhost:9002/segment/simple" \
  -F "file=@/path/to/test/image.jpg"
```

---

## 🎯 Your First Hour Goals

### Hour 1: Get APIs Running
- ✅ API keys configured
- ✅ Dependencies installed
- ✅ At least ONE service running
- ✅ Test request succeeds

### Success Criteria
```bash
# This should work:
curl http://localhost:9001  # GeoGPT
# OR
curl http://localhost:9002  # AutoLabel
```

---

## 🔥 Next Steps (Hour 2-4)

### For GeoGPT Track:
1. Add actual function calling (fly camera, load layers)
2. Create a simple chat UI in DeepGIS frontend
3. Test voice input with Whisper

### For AutoLabel Track:
1. Download SAM model weights
2. Integrate SAM inference
3. Create UI for click-to-segment

### For Other Tracks:
1. Fork one of the minimal examples above
2. Customize for your track's needs
3. Start building core functionality

---

## 📚 Essential Reading (15 minutes)

### Before You Code:
1. **Read:** `THANKSGIVING_2025_AI_HACKATHON_PLAN.md` (your track's section)
2. **Skim:** `DEEPGIS_XR_URL_PATTERN_ANALYSIS.md` (understand the existing architecture)
3. **Review:** `deepgis-xr/deepgis_xr/apps/web/urls.py` (see existing API endpoints)

### API Documentation:
- OpenAI: https://platform.openai.com/docs
- LangChain: https://python.langchain.com/docs
- FastAPI: https://fastapi.tiangolo.com/

---

## 🐛 Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'openai'"
**Fix:** 
```bash
source venv/bin/activate
pip install openai
```

### Issue: "OpenAI API key invalid"
**Fix:**
```bash
# Check your key
cat .env.ai | grep OPENAI_API_KEY

# Make sure it starts with sk-
# Get a new key from: https://platform.openai.com/api-keys
```

### Issue: "Connection refused on port 9001"
**Fix:**
```bash
# Make sure the service is running
ps aux | grep python | grep geogpt

# Check if port is in use
netstat -tulpn | grep 9001

# Kill any conflicting process
pkill -f "geogpt"
```

### Issue: "CORS error in browser"
**Fix:** Already handled in the minimal examples with `CORSMiddleware`

---

## 📞 Getting Help

### During Hackathon:
- **Discord:** #ai-hackathon channel
- **In-Person:** Find track leads
- **Docs:** This folder has everything you need

### Debugging Checklist:
1. ✅ Is the service running? (`ps aux | grep python`)
2. ✅ Can you curl it? (`curl http://localhost:9001`)
3. ✅ Are API keys set? (`echo $OPENAI_API_KEY`)
4. ✅ Check logs for errors
5. ✅ Ask for help! (don't waste time stuck)

---

## 🎉 You're Ready!

Now go to your assigned track in `THANKSGIVING_2025_AI_HACKATHON_PLAN.md` and start building! 🚀

**Remember:**
- ✅ MVP first, polish later
- ✅ Commit often
- ✅ Demo early, demo often
- ✅ Have fun! 🦃🤖

---

**Good luck and happy hacking! 🎉**

