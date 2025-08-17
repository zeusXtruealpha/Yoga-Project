const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const PORT = 3001;
const FLASK_BASE = "http://127.0.0.1:5000";  

app.use(cors());

// Multer will store file in memory
const storage = multer.memoryStorage();
const upload = multer({ storage });


app.get('/api/health', async (req, res) => {
  res.json({ node: "ok" });
});

// Person detection route
app.post('/api/check_person', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    console.log(">>> Forwarding to", `${FLASK_BASE}/check_person`);

    const response = await axios.post(`${FLASK_BASE}/check_person`, form, {
      headers: form.getHeaders()
    });

    res.json(response.data);
  } catch (error) {
    console.error('[ERROR] Person detection failed:', error.message);
    res.status(500).json({ error: 'Person detection failed' });
  }
});

// Mood prediction route
app.post('/api/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log(`[INFO] Received image: ${req.file.originalname} (${req.file.size} bytes)`);

    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    console.log(">>> Forwarding to", `${FLASK_BASE}/predict`);

    const response = await axios.post(`${FLASK_BASE}/predict`, form, {
      headers: form.getHeaders()
    });

    res.json(response.data);
  } catch (error) {
    console.error('[ERROR] Prediction failed:', error.message);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

app.listen(PORT, () => {
  console.log(` API server running on http://localhost:${PORT}`);
});
