# CLIP Search Query Limitations & Best Practices

## Summary

CLIP understands **visual descriptions** of things it has seen during training (internet images with text). It struggles with abstract concepts, technical jargon, negations, and specific quantities.

---

## ✅ What Works Well

### 1. Visual Descriptions
```
"smooth surface"
"rough texture"  
"shiny crystalline"
"layered structure"
"fractured rock"
```

### 2. Colors
```
"white quartz"
"dark grey basalt"
"orange weathered surface"
"greenish mineral"
"red iron-rich rock"
```

### 3. Common Objects/Materials
```
"granite"
"sandstone"
"crystal"
"stone"
"rock"
```

### 4. Relative Sizes
```
"large crystals"
"fine-grained texture"
"coarse surface"
"small pores"
```

### 5. Combined Features
```
"grey rock with white crystals"
"smooth dark volcanic stone"
"rough orange weathered surface"
```

---

## ❌ What Doesn't Work

### 1. Technical/Scientific Terms
**Bad:**
```
"plagioclase feldspar phenocrysts"
"mafic igneous composition"
"greenschist facies metamorphism"
```

**Better:**
```
"white crystal minerals"
"dark volcanic rock"
"layered foliated rock"
```

### 2. Abstract Concepts
**Bad:**
```
"ancient rock formation"
"valuable mineral"
"rare specimen"
"high-quality sample"
```

**Better:**
```
"weathered eroded surface"
"crystalline mineral"
"unusual texture"
"well-preserved surface"
```

### 3. Negations
**Bad:**
```
"not weathered"
"without crystals"
"no visible layers"
```

**Better:**
```
"fresh unweathered surface"
"smooth uniform texture"
"massive structure"
```

### 4. Specific Quantities
**Bad:**
```
"exactly 3 large crystals"
"50% quartz content"
"2cm diameter minerals"
"5mm grain size"
```

**Better:**
```
"several large crystals"
"mostly quartz"
"large mineral grains"
"fine grain size"
```

### 5. Overly Complex Queries
**Bad:**
```
"granite with large orthoclase feldspar phenocrysts, 
minor quartz, biotite flakes, showing advanced 
weathering with iron oxidation producing orange patina"
```

**Better (multiple simpler queries):**
```
1. "granite with large crystals"
2. "rock with dark mineral flakes"  
3. "orange weathered surface"
```

### 6. Temporal/Process Descriptions
**Bad:**
```
"recently formed crystals"
"slowly weathered over centuries"
"rapidly cooled volcanic rock"
```

**Better:**
```
"fresh crystal growth"
"heavily weathered surface"
"fine-grained volcanic texture"
```

---

## 📊 Performance by Category

| Query Type | Performance | Examples |
|------------|-------------|----------|
| Color | ⭐⭐⭐⭐⭐ | "red rock", "white crystals" |
| Texture | ⭐⭐⭐⭐⭐ | "smooth", "rough", "crystalline" |
| Common Terms | ⭐⭐⭐⭐ | "granite", "sandstone", "basalt" |
| Relative Size | ⭐⭐⭐⭐ | "large", "fine-grained", "coarse" |
| Visual Features | ⭐⭐⭐⭐ | "layered", "fractured", "porous" |
| Rock Types | ⭐⭐⭐ | "igneous", "sedimentary", "volcanic" |
| Technical Terms | ⭐⭐ | "feldspar", "vesicular", "foliation" |
| Very Technical | ⭐ | "plagioclase", "greenschist facies" |
| Abstract Concepts | ⭐ | "valuable", "ancient", "rare" |
| Negations | ❌ | "not X", "without Y" |
| Quantities | ❌ | "3 crystals", "50% quartz" |

---

## 🎯 Optimization Strategies

### Strategy 1: Iterate from Broad to Specific

```bash
# Start broad
"crystalline rock"

# Get more specific based on results
"rock with large white crystals"

# Add more detail if needed
"granite with white quartz crystals"
```

### Strategy 2: Use Multiple Queries

Instead of one complex query, try several simple ones:

```bash
# Don't do this:
"grey granite with large white feldspar, dark biotite, weathered orange"

# Do this instead:
Query 1: "grey granite texture"
Query 2: "rock with white crystals"
Query 3: "orange weathered surface"
Query 4: "rock with dark minerals"
```

### Strategy 3: Think Like an Image Caption

CLIP was trained on image captions. Write queries like you're captioning a photo:

**Good (caption-like):**
- "smooth polished stone surface"
- "rough fractured grey rock"
- "crystalline texture with sparkles"

**Bad (not caption-like):**
- "Type II quartz with fluid inclusions"
- "Sample #1234 from drill core"
- "High metamorphic grade specimen"

### Strategy 4: Use Common Language

Pretend you're describing the rock to a non-geologist:

**Good:**
- "rock with visible crystals"
- "layered striped pattern"
- "bubbly volcanic texture"

**Bad:**
- "porphyritic texture"
- "xenolithic inclusions"
- "aphanitic groundmass"

---

## 🧪 Test Your Queries

Run this to test different query types:

```bash
chmod +x test_clip_queries.sh
./test_clip_queries.sh
```

Then visually inspect results to see what works best for your dataset!

---

## 💡 Tips for Rock Tiles Specifically

### Your 8,083 Rock Tiles Dataset

**What likely works well:**
1. Color variations (grey, white, orange, red, etc.)
2. Texture differences (smooth vs rough, fine vs coarse)
3. Visible features (crystals, layers, pores, fractures)
4. General rock types (granite, sandstone, basalt)

**What might not work:**
1. Specific mineral names (unless very common)
2. Technical geological classifications
3. Microscale features (too small to see in 256×256 tiles)
4. Abstract geological concepts

### Recommended Query Approach:

1. **Start with simple visual features:**
   - Colors: "dark rock", "light stone", "orange surface"
   - Textures: "smooth", "rough", "crystalline"

2. **Add rock types if known:**
   - "granite texture"
   - "sandstone layers"
   - "volcanic rock"

3. **Combine features:**
   - "grey granite with white crystals"
   - "rough orange weathered surface"

4. **Iterate based on results:**
   - Look at what comes back
   - Refine your query
   - Try variations

---

## 🔬 Understanding Similarity Scores

Typical score ranges for your dataset:
- **0.35-0.40**: Excellent match (rare)
- **0.30-0.35**: Good match (typical for top results)
- **0.25-0.30**: Moderate match
- **0.20-0.25**: Weak match
- **<0.20**: Poor match

**Note:** Rock tiles are visually similar, so scores tend to be lower than diverse datasets. A score of 0.33 is actually quite good!

---

## 📝 Query Examples by Category

### Texture Queries
```bash
python3 vlm_clip_simple.py --mode search --query "smooth polished surface"
python3 vlm_clip_simple.py --mode search --query "rough fractured rock"
python3 vlm_clip_simple.py --mode search --query "crystalline sparkly texture"
python3 vlm_clip_simple.py --mode search --query "fine-grained smooth surface"
python3 vlm_clip_simple.py --mode search --query "coarse granular texture"
```

### Color Queries
```bash
python3 vlm_clip_simple.py --mode search --query "white rock"
python3 vlm_clip_simple.py --mode search --query "dark grey stone"
python3 vlm_clip_simple.py --mode search --query "orange weathered surface"
python3 vlm_clip_simple.py --mode search --query "reddish iron-rich rock"
python3 vlm_clip_simple.py --mode search --query "light colored granite"
```

### Feature Queries
```bash
python3 vlm_clip_simple.py --mode search --query "visible crystals"
python3 vlm_clip_simple.py --mode search --query "layered structure"
python3 vlm_clip_simple.py --mode search --query "fractured surface"
python3 vlm_clip_simple.py --mode search --query "porous texture"
python3 vlm_clip_simple.py --mode search --query "weathered eroded surface"
```

### Combined Queries
```bash
python3 vlm_clip_simple.py --mode search --query "grey rock with white crystals"
python3 vlm_clip_simple.py --mode search --query "smooth dark volcanic stone"
python3 vlm_clip_simple.py --mode search --query "rough orange weathered granite"
python3 vlm_clip_simple.py --mode search --query "fine-grained grey rock"
```

---

## 🚀 Next Steps

1. **Experiment** - Try different query styles
2. **Visual Verification** - Check the saved images
3. **Iterate** - Refine based on what works
4. **Document** - Note which queries work well for your dataset

Remember: CLIP is powerful but has limitations. The key is understanding what it can and cannot do, then crafting queries accordingly!

