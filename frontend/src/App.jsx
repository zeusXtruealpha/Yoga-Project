import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [mood, setMood] = useState("");
  const [feedback, setFeedback] = useState([]);

  const [loadingMood, setLoadingMood] = useState(false);
  const [loadingPerson, setLoadingPerson] = useState(false);

  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target.result);
      setMood("");
      setFeedback([]);
    };
    reader.readAsDataURL(file);
  };

  const uploadImageToNode = async (endpoint) => {
    const blob = await fetch(image).then((r) => r.blob());
    const formData = new FormData();
    formData.append("file", blob, "face.jpg");

    const response = await fetch(`http://localhost:3001/api/${endpoint}`, {
      method: "POST",
      body: formData,
    });

    const cloned = response.clone();

    let data;
    try {
      data = await response.json();
    } catch (err) {
      const errorText = await cloned.text();
      console.error("❌ Backend returned non-JSON:", errorText);
      throw new Error(`Invalid JSON (HTTP ${response.status})`);
    }

    if (!response.ok) {
      throw new Error(data.error || `Server error (HTTP ${response.status})`);
    }

    return data;
  };

  const handlePredictMood = async () => {
    if (!image) return;
    setLoadingMood(true);

    try {
      const data = await uploadImageToNode("predict");
      console.log("✅ Mood detection result:", data);
      setMood(data.mood || "Unknown");
    } catch (error) {
      console.error("❌ Mood detection failed:", error);
      alert("Mood detection failed: " + error.message);
    } finally {
      setLoadingMood(false);
    }
  };

  const handleCheckPerson = async () => {
    if (!image) return;
    setLoadingPerson(true);

    try {
      const data = await uploadImageToNode("check_person");
      console.log("✅ Person detection result:", data);

      if (!data.person_detected) {
        setFeedback(["No person detected"]);
        alert("⚠️ No person detected!");
      } else {
        setFeedback(data.feedback || ["Aligned properly"]);
      }
    } catch (error) {
      console.error("❌ Person detection failed:", error);
      alert("Person detection failed: " + error.message);
    } finally {
      setLoadingPerson(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Face Mood & Human Detection</h1>
        <p>Upload your photo to detect your mood or check human presence</p>
      </header>

      <main className="main-content">
        {image ? (
          <div className="image-preview">
            <img src={image} alt="Preview" />
            <button
              onClick={() => {
                setImage(null);
                setMood("");
                setFeedback([]);
              }}
              className="remove-btn"
            >
              Remove
            </button>
          </div>
        ) : (
          <div
            className="upload-area"
            onClick={() => fileInputRef.current.click()}
          >
            <span>Click to upload image</span>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageChange}
              accept="image/*"
              hidden
            />
          </div>
        )}

        <div className="button-container">
          <button
            onClick={handlePredictMood}
            disabled={!image || loadingMood}
            className={`predict-btn ${!image || loadingMood ? "disabled" : ""}`}
          >
            {loadingMood ? "Analyzing..." : "Detect Mood"}
          </button>

          <button
            onClick={handleCheckPerson}
            disabled={!image || loadingPerson}
            className={`predict-btn ${!image || loadingPerson ? "disabled" : ""}`}
          >
            {loadingPerson ? "Checking..." : "Check Human Presence"}
          </button>
        </div>

        {mood && (
          <div className="result-container">
            <h2>You look {mood}!</h2>
            <p>Detected mood: {mood}</p>
          </div>
        )}

        {feedback.length > 0 && (
          <div className="result-container">
            <h2>Human Detection Feedback</h2>
            <ul>
              {feedback.map((msg, idx) => (
                <li key={idx}>{msg}</li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
