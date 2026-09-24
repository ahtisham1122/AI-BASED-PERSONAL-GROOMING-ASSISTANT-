# AI Grooming Assistant — Frontend Prototype

An elegant, framework-free single-page web application (SPA) prototype designed for academic defense presentations and AI model integration.

The application allows users to select their gender and occasion, upload a front-facing portrait photo, and receive simulated AI recommendations including:
- **Face Shape Landmark Detection** (e.g., Oval, Square, Round, Heart)
- **Skin Tone Classification** (e.g., Warm, Cool, Neutral, Deep)
- **Eyewear Style Recommendations** & Interactive Try-On Preview
- **Wardrobe Color Palette Swatches** tailored to analyzed skin undertones
- **Tailored Outfit Recommendations** looked up dynamically across 24 distinct style combinations

---

## 📁 File Structure

```
ai-grooming-assistant/
│
├── index.html          # Single Page Application HTML structure
├── style.css           # Vanilla CSS3 design system, tokens & responsive layouts
├── script.js           # State management, validation, mock API simulator & DOM renderer
│
├── assets/
│   ├── glasses/        # Visual glasses frame assets (rectangle, round, wayfarer, aviator)
│   └── icons/          # SVG icons
│
└── README.md           # Project documentation & integration guide
```

---

## 🚀 Running the Project

Since the project is built with standard **HTML5**, **CSS3**, and **Vanilla JavaScript**, no installation or compilation step is required!

### Option 1: Direct File Access
Simply open `index.html` directly in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Safari):
```bash
# Double click index.html or open via terminal
```

### Option 2: Local HTTP Server (Recommended)
Using a lightweight local HTTP server:
```bash
# Using Node.js npx serve
npx -y serve .

# Or using Python 3
python -m http.server 8000
```
Then navigate to `http://localhost:8000` or `http://localhost:3000` in your web browser.

---

## 🧪 Demo Mode

The current version operates in **Demo Mode**. The frontend simulates the multi-step AI analysis with realistic loading step indicators and returns structured mock analysis data.

1. Select **Gender** (*Men* or *Women*).
2. Select **Occasion** (*Casual*, *Formal*, or *Party*).
3. Upload a sample portrait image (`.jpg` or `.png`).
4. Click **Analyze My Look**.
5. Observe the loading state and explore the generated results report!
6. Click **Try Another Photo** to reset all state cleanly.

---

## 🔌 Future AI Backend / API Integration

To connect this frontend to a real AI backend model (e.g., Python FastAPI/Flask or Node.js server running OpenCV / PyTorch / MediaPipe face analysis):

Open `script.js` and locate the `analyzeImage()` function (Line 385):

```javascript
async function analyzeImage(file, gender, occasion) {
  // =========================================
  // FUTURE BACKEND/API INTEGRATION POINT
  // =========================================
  
  const formData = new FormData();
  formData.append('image', file);
  formData.append('gender', gender);
  formData.append('occasion', occasion);

  const response = await fetch('https://your-api-domain.com/api/v1/analyze', {
      method: 'POST',
      body: formData
  });

  if (!response.ok) {
    throw new Error('API analysis failed');
  }

  return await response.json();
}
```

### Expected API Response Format

Your API endpoint should return a JSON response formatted as follows:

```json
{
  "faceShape": "Oval",
  "skinTone": "Warm",
  "confidence": 92,
  "glasses": [
    {
      "name": "Rectangle Frames",
      "image": "assets/glasses/rectangle.svg",
      "reason": "Structured frames complement the balanced proportions of an oval face."
    },
    {
      "name": "Round Frames",
      "image": "assets/glasses/round.svg",
      "reason": "Round frames create a softer contrast with your facial structure."
    },
    {
      "name": "Wayfarer Frames",
      "image": "assets/glasses/wayfarer.svg",
      "reason": "Wayfarer frames add definition while maintaining balance."
    },
    {
      "name": "Aviator Frames",
      "image": "assets/glasses/aviator.svg",
      "reason": "Aviator frames add a stylish contrast to your facial proportions."
    }
  ]
}
```

---

## 🛠️ Data Configuration Guide

All recommendation datasets are maintained centrally in `script.js`:

- **Color Palettes**: Modify `colorPalettes` object to adjust hex codes, swatch names, or explanations for `Warm`, `Cool`, `Neutral`, and `Deep` skin tones.
- **Outfit Matrix**: Modify `outfitRecommendations` object to update wardrobe text recommendations across Gender × Occasion × Color Group combinations.
- **Face Shape Explanations**: Update `faceShapeDetails` object to edit face shape descriptions.

---

## 📋 Academic Presentation Notes
- Target screen resolution: Optimized for standard laptop presentation displays (**1366×768** and **1440×900**).
- Zero external framework dependencies ensure high performance and immediate page load without network latency.
