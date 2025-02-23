document.addEventListener("DOMContentLoaded", function () {
    const uploadInput = document.getElementById("upload");
    const clearImageBtn = document.getElementById("clearImage");
    const predictImageBtn = document.getElementById("predictImage");
    const uploadedImage = document.getElementById("uploadedImage");
    const resultImage = document.getElementById("resultImage");

    const canvas = document.getElementById("canvas");
    const clearCanvasBtn = document.getElementById("clearCanvas");
    const predictCanvasBtn = document.getElementById("predictCanvas"); // ✅ This line ensures it's found
    const resultCanvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");

    // Set canvas size
    canvas.width = 280;
    canvas.height = 280;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    let drawing = false;

    // Drawing functionality
    canvas.addEventListener("mousedown", () => (drawing = true));
    canvas.addEventListener("mouseup", () => {
        drawing = false;
        ctx.beginPath();
    });
    canvas.addEventListener("mousemove", (event) => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    });

    // Clear Canvas
    clearCanvasBtn.addEventListener("click", () => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultCanvas.textContent = "";
    });

    // Predict from Canvas
    predictCanvasBtn.addEventListener("click", function () {
        const canvasData = canvas.toDataURL("image/png");

        fetch("/predict_canvas", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: canvasData }),
        })
            .then((response) => response.json())
            .then((data) => {
                resultCanvas.textContent = `Prediction: ${data.prediction}`;
            })
            .catch((error) => console.error("Error:", error));
    });

    // Image Upload Preview
    uploadInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = "block"; // Show image
            };
            reader.readAsDataURL(file);
        }
    });

    // Clear Image
    clearImageBtn.addEventListener("click", function () {
        uploadInput.value = "";
        uploadedImage.src = "";
        uploadedImage.style.display = "none";
        resultImage.textContent = "";
    });

    // Predict Uploaded Image
    predictImageBtn.addEventListener("click", function () {
        const file = uploadInput.files[0];
        if (!file) {
            resultImage.textContent = "Please upload an image first!";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        fetch("/predict_image", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                resultImage.textContent = `Prediction: ${data.prediction}`;
            })
            .catch((error) => console.error("Error:", error));
    });
});
