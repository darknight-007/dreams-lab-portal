# 🦃 Thanksgiving AI Hackathon - Team Assignments

**Event:** November 27-30, 2025  
**Location:** DeepGIS Lab / Remote  
**Total Participants:** 15 developers

---

## 🎯 Track Assignments

### Track 1: 🤖 GeoGPT (Natural Language Interface)
**Priority:** 🔥 Critical  
**Team Size:** 4 people  
**Port:** 9001

#### Team Members
- **Track Lead:** _________________ (LLM Engineer)
  - **Responsibilities:** Architecture, OpenAI integration, prompt engineering
  - **Focus:** Chat engine, function calling, context management
  
- **Backend Developer:** _________________ (Python/FastAPI)
  - **Responsibilities:** API endpoints, WebSocket, database integration
  - **Focus:** FastAPI routes, DeepGIS API integration
  
- **Frontend Developer:** _________________ (JavaScript/TypeScript)
  - **Responsibilities:** Chat UI, DeepGIS integration, WebSocket client
  - **Focus:** Chat widget, visual feedback, command visualization
  
- **Voice Engineer:** _________________ (Audio Processing)
  - **Responsibilities:** Whisper integration, TTS, audio pipeline
  - **Focus:** Speech-to-text, text-to-speech, audio preprocessing

#### Daily Goals
- **Day 1:** Basic chat working, 5 commands implemented
- **Day 2:** Function calling working, 15 commands, voice input
- **Day 3:** Context management, conversation history, voice output
- **Day 4:** Polish, demo prep, edge case handling

#### Success Metrics
- ✅ 20+ natural language commands working
- ✅ 90%+ intent classification accuracy
- ✅ Response time < 2 seconds
- ✅ Voice input works reliably

---

### Track 2: 👁️ AutoLabel AI (Computer Vision)
**Priority:** 🔥 Critical  
**Team Size:** 4 people  
**Port:** 9002

#### Team Members
- **Track Lead:** _________________ (Computer Vision Engineer)
  - **Responsibilities:** Model selection, SAM integration, CV pipeline
  - **Focus:** Segmentation algorithms, model optimization
  
- **ML Engineer:** _________________ (Model Training)
  - **Responsibilities:** Model fine-tuning, ONNX conversion, optimization
  - **Focus:** Performance tuning, quantization, model deployment
  
- **Backend Developer:** _________________ (Python/PyTorch)
  - **Responsibilities:** Inference API, preprocessing, postprocessing
  - **Focus:** FastAPI inference endpoints, image processing
  
- **Frontend Developer:** _________________ (Visualization)
  - **Responsibilities:** Annotation UI, mask overlay, user feedback
  - **Focus:** Cesium overlay rendering, drawing tools

#### Daily Goals
- **Day 1:** SAM model deployed, basic segmentation API
- **Day 2:** Object detection working, change detection prototype
- **Day 3:** 3D building extraction, NDVI analysis
- **Day 4:** UI polish, performance optimization, demo

#### Success Metrics
- ✅ SAM model deployed and accessible
- ✅ Object detection accuracy > 85%
- ✅ Change detection works on test dataset
- ✅ Inference time < 5 seconds

---

### Track 3: 🧠 ForecastGIS (Predictive Analytics)
**Priority:** 🟡 High  
**Team Size:** 3 people  
**Port:** 9003

#### Team Members
- **Track Lead:** _________________ (Data Scientist)
  - **Responsibilities:** Algorithm design, model selection, validation
  - **Focus:** Path planning, temporal prediction, anomaly detection
  
- **ML Engineer:** _________________ (Time Series & Optimization)
  - **Responsibilities:** Model training, optimization algorithms
  - **Focus:** LSTM/XGBoost training, OR-Tools optimization
  
- **Backend Developer:** _________________ (Python)
  - **Responsibilities:** API implementation, data pipeline
  - **Focus:** FastAPI endpoints, data preprocessing, caching

#### Daily Goals
- **Day 1:** Path planning algorithm, spatial sampling optimizer
- **Day 2:** Temporal prediction model, anomaly detection
- **Day 3:** Recommendation engine, integration with frontend
- **Day 4:** Model tuning, confidence intervals, demo

#### Success Metrics
- ✅ Path planning generates valid routes
- ✅ Temporal predictions show reasonable accuracy
- ✅ Anomaly detection catches known issues
- ✅ Recommendations are relevant

---

### Track 4: 🎙️ HandsFree GIS (Voice & Multimodal)
**Priority:** 🟡 High  
**Team Size:** 2 people  
**Port:** 9004

#### Team Members
- **Track Lead:** _________________ (Audio Engineer)
  - **Responsibilities:** Voice pipeline, Whisper integration, wake word
  - **Focus:** Speech recognition, audio processing, command parsing
  
- **Frontend Developer:** _________________ (UI/UX)
  - **Responsibilities:** Voice UI, visual feedback, gesture integration
  - **Focus:** Voice overlay, listening states, audio feedback

#### Daily Goals
- **Day 1:** Whisper integration, basic commands working
- **Day 2:** Core navigation commands, TTS responses
- **Day 3:** Wake word detection, voice annotation
- **Day 4:** UI polish, multimodal integration, demo

#### Success Metrics
- ✅ Voice recognition accuracy > 90%
- ✅ 20+ voice commands working
- ✅ Response latency < 1 second
- ✅ Works with push-to-talk

---

### Track 5: 🔍 GeoSearch Pro (Semantic Search)
**Priority:** 🟢 Medium  
**Team Size:** 2 people  
**Port:** 9005

#### Team Members
- **Track Lead:** _________________ (Search Engineer)
  - **Responsibilities:** CLIP integration, embedding pipeline, search algorithm
  - **Focus:** Semantic search, visual search, ranking
  
- **Backend Developer:** _________________ (Vector Database)
  - **Responsibilities:** Vector DB setup, API implementation, indexing
  - **Focus:** Weaviate/Pinecone integration, query optimization

#### Daily Goals
- **Day 1:** CLIP model setup, embedding generation
- **Day 2:** Semantic search API, visual search
- **Day 3:** Metadata enrichment, smart filtering
- **Day 4:** Search UI, result previews, demo

#### Success Metrics
- ✅ Semantic search returns relevant results
- ✅ Visual search works with uploaded images
- ✅ Search latency < 500ms
- ✅ Metadata enrichment adds value

---

## 🏗️ Infrastructure Team (Shared Responsibility)

### AI Gateway
**Owner:** Track 1 Lead (with help from all)
- Request routing
- Rate limiting
- Authentication
- Response caching

### Vector Database
**Owner:** Track 5 Lead
- ChromaDB/Weaviate setup
- Embedding management
- Backup strategy

### Monitoring & Logging
**Owner:** Rotating daily responsibility
- Prometheus setup
- Log aggregation
- Performance metrics
- Error tracking

---

## 📅 Daily Standups

### Format (15 minutes each)
**Time:** 9:00 AM every day

**Round-robin updates:**
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers?
4. Help needed?

**Track Leads:**
- Share progress screenshots
- Highlight integration points
- Request cross-team help

---

## 🤝 Integration Points

### Track 1 (GeoGPT) ↔ Track 2 (AutoLabel)
**Integration:** GeoGPT calls AutoLabel for segmentation
```
User: "Detect all buildings in this area"
→ GeoGPT → AutoLabel API → Return results → GeoGPT displays
```

### Track 1 (GeoGPT) ↔ Track 3 (ForecastGIS)
**Integration:** GeoGPT calls ForecastGIS for predictions
```
User: "Predict vegetation changes next month"
→ GeoGPT → ForecastGIS API → Return prediction → GeoGPT visualizes
```

### Track 1 (GeoGPT) ↔ Track 4 (HandsFree)
**Integration:** Voice → GeoGPT → Action
```
Voice: "Show me Mount Everest"
→ HandsFree (STT) → GeoGPT (understand) → DeepGIS (action)
```

### Track 1 (GeoGPT) ↔ Track 5 (GeoSearch)
**Integration:** GeoGPT uses semantic search
```
User: "Find imagery with urban development"
→ GeoGPT → GeoSearch API → Return matches → GeoGPT loads layer
```

### All Tracks ↔ DeepGIS Frontend
**Integration:** Each track exposes REST/WebSocket APIs
- Frontend calls AI services via AI Gateway (port 9000)
- Real-time updates via WebSocket
- Async operations with status polling

---

## 🔧 Development Environment Setup

### Shared Resources

#### Git Repository
```bash
# Main branch: main
# Feature branches: track-1/feature-name, track-2/feature-name, etc.

# Branch naming convention:
track-1/geogpt-chat-engine
track-2/sam-integration
track-3/path-planning
track-4/voice-commands
track-5/semantic-search
```

#### Code Review Process
- All PRs require 1 approval from track lead
- Integration PRs require 2 approvals (both track leads)
- Use GitHub PR templates
- Daily merge to `dev` branch at 5 PM

#### Environment Variables
```bash
# Shared .env.ai file (DO NOT COMMIT)
# Each developer gets a copy

# Location: deepgis-xr/.env.ai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
HUGGINGFACE_TOKEN=hf_...
```

#### Port Allocation (Reminder)
- **9000:** AI Gateway (all traffic goes here first)
- **9001:** GeoGPT
- **9002:** AutoLabel AI
- **9003:** ForecastGIS
- **9004:** HandsFree
- **9005:** GeoSearch
- **9100:** ChromaDB
- **9200:** Weaviate
- **6379:** Redis

---

## 📊 Daily Progress Tracking

### Day 1 Checklist (All Tracks)
- [ ] Environment setup complete
- [ ] Basic API endpoint responding
- [ ] First integration test passing
- [ ] Documentation started

### Day 2 Checklist (All Tracks)
- [ ] Core features implemented
- [ ] Cross-track integration working
- [ ] Unit tests written
- [ ] Mid-hackathon demo prepared

### Day 3 Checklist (All Tracks)
- [ ] Feature complete
- [ ] UI/UX polished
- [ ] Performance optimized
- [ ] Documentation updated

### Day 4 Checklist (All Tracks)
- [ ] Demo script ready
- [ ] Backup video recorded
- [ ] Code reviewed and merged
- [ ] Presentation slides prepared

---

## 🎬 Demo Coordination

### Demo Order (Day 4, 2:30 PM)
1. **Opening** (2 min) - Project overview
2. **Track 1: GeoGPT** (10 min) - Natural language magic
3. **Track 2: AutoLabel** (10 min) - Computer vision wow
4. **Track 3: ForecastGIS** (8 min) - Predictive power
5. **Track 4: HandsFree** (8 min) - Voice control showcase
6. **Track 5: GeoSearch** (8 min) - Semantic search demo
7. **Integration Demo** (10 min) - All tracks working together
8. **Q&A** (10 min)
9. **Closing** (5 min) - What's next, thank you

### Demo Script Template (Each Track)
```
1. Problem Statement (1 min)
   "Currently, users have to manually..."

2. Solution Introduction (1 min)
   "We built an AI that can..."

3. Live Demo (5-7 min)
   - Show 3-5 key features
   - Emphasize wow moments
   - Have backup video ready

4. Technical Highlights (1 min)
   "Under the hood, we used..."

5. Future Work (30 sec)
   "Next steps include..."
```

---

## 💬 Communication Channels

### Discord Channels
- **#general** - Announcements, general discussion
- **#track-1-geogpt** - GeoGPT team
- **#track-2-autolabel** - AutoLabel team
- **#track-3-forecast** - ForecastGIS team
- **#track-4-handsfree** - HandsFree team
- **#track-5-geosearch** - GeoSearch team
- **#integration** - Cross-track coordination
- **#random** - Memes, food, fun

### Video Calls
- **Daily Standups:** Zoom/Google Meet (9:00 AM)
- **Integration Meetings:** As needed
- **Emergency Help:** Ping in Discord, call if urgent

### Documentation
- **Shared Google Doc:** Meeting notes, decisions
- **GitHub Wiki:** Technical documentation
- **Notion/Confluence:** Design documents (if available)

---

## 🍕 Logistics

### Food & Breaks
- **Coffee/Snacks:** Available all day
- **Lunch:** 12:00-1:00 PM (provided)
- **Dinner (Day 1):** Thanksgiving dinner 🦃
- **Breaks:** Take them! Don't burn out.

### Working Hours (Flexible)
- **Core Hours:** 10 AM - 5 PM (everyone available)
- **Late Night:** Optional, but track lead should be around
- **Day 4:** 9 AM - 4 PM (hard stop for demo)

### Remote Participation
- All meetings recorded
- Async updates in Discord
- Screen sharing for remote demos

---

## 🏆 Awards & Recognition

### Categories
1. **Best Innovation** - Most creative use of AI
2. **Most Impressive Demo** - Wow factor
3. **Best Code Quality** - Clean, tested, documented
4. **Best Teamwork** - Collaboration across tracks
5. **People's Choice** - Team vote

### Prizes
- 🥇 Gold: $500 gift card + trophy
- 🥈 Silver: $300 gift card + certificate
- 🥉 Bronze: $200 gift card + certificate
- 🎉 All participants: Hackathon swag + certificate

---

## 📝 Final Deliverables (Due Day 4, 2:00 PM)

### Code
- [ ] All code merged to `main` branch
- [ ] README updated with setup instructions
- [ ] Environment variables documented
- [ ] Docker Compose tested and working

### Documentation
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Architecture diagram
- [ ] Setup guide
- [ ] User guide (for demo)

### Demo Materials
- [ ] Presentation slides (10 min)
- [ ] Live demo script
- [ ] Backup video (in case live demo fails)
- [ ] Example queries/inputs

### Testing
- [ ] Unit tests written
- [ ] Integration tests passing
- [ ] Manual testing completed
- [ ] Performance benchmarks documented

---

## 🚨 Emergency Contacts

### Track Leads
- **Track 1 (GeoGPT):** _______________ - Phone: _______________
- **Track 2 (AutoLabel):** _______________ - Phone: _______________
- **Track 3 (ForecastGIS):** _______________ - Phone: _______________
- **Track 4 (HandsFree):** _______________ - Phone: _______________
- **Track 5 (GeoSearch):** _______________ - Phone: _______________

### Organizers
- **Project Lead:** _______________ - Phone: _______________
- **Tech Support:** _______________ - Phone: _______________
- **Facilities:** _______________ - Phone: _______________

### Emergency
- **Building Security:** _______________
- **IT Helpdesk:** _______________
- **Medical:** 911

---

## ✅ Pre-Hackathon Preparation (Each Team Member)

### By November 26 (Day Before)
- [ ] Read hackathon plan thoroughly
- [ ] Review your track's section
- [ ] Install required software
- [ ] Test API keys
- [ ] Clone repository
- [ ] Set up development environment
- [ ] Introduce yourself in Discord
- [ ] Review integration points with other tracks

### What to Bring
- [ ] Laptop (fully charged)
- [ ] Charger and cables
- [ ] Headphones (for focus/calls)
- [ ] Water bottle
- [ ] Positive attitude! 🎉

---

## 🎯 Team Agreements

### We Commit To:
1. ✅ **Communication:** Keep team updated, ask for help when stuck
2. ✅ **Respect:** Value everyone's ideas, be kind and patient
3. ✅ **Quality:** Write clean, tested, documented code
4. ✅ **Integration:** Make APIs easy for others to use
5. ✅ **Fun:** Enjoy the process, celebrate small wins! 🎉

### Conflict Resolution:
- Talk it out first
- Escalate to track lead if needed
- Project lead makes final decisions

---

## 📖 Resources Quick Links

### Hackathon Docs
- [Full Hackathon Plan](./THANKSGIVING_2025_AI_HACKATHON_PLAN.md)
- [Quick Start Guide](./HACKATHON_QUICKSTART.md)
- [Architecture Analysis](./DEEPGIS_XR_URL_PATTERN_ANALYSIS.md)

### External Docs
- [OpenAI API](https://platform.openai.com/docs)
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Cesium](https://cesium.com/learn/)
- [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything)

### Learning Materials
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [WebSocket Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

---

**Let's build something amazing together! 🚀🦃🤖**

---

## Signature

I have read and agree to the above team assignments and commitments:

**Track 1 Members:**
- _________________ (Lead) - Date: _______
- _________________ (Backend) - Date: _______
- _________________ (Frontend) - Date: _______
- _________________ (Voice) - Date: _______

**Track 2 Members:**
- _________________ (Lead) - Date: _______
- _________________ (ML) - Date: _______
- _________________ (Backend) - Date: _______
- _________________ (Frontend) - Date: _______

**Track 3 Members:**
- _________________ (Lead) - Date: _______
- _________________ (ML) - Date: _______
- _________________ (Backend) - Date: _______

**Track 4 Members:**
- _________________ (Lead) - Date: _______
- _________________ (Frontend) - Date: _______

**Track 5 Members:**
- _________________ (Lead) - Date: _______
- _________________ (Backend) - Date: _______

**Project Lead:** _________________ - Date: _______

---

**Document Version:** 1.0  
**Created:** November 27, 2025  
**Last Updated:** November 27, 2025

