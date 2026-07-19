document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const canvasWrapper = document.querySelector('.canvas-wrapper');
    
    const btnUndo = document.getElementById('btn-undo');
    const btnClear = document.getElementById('btn-clear');
    const btnPredict = document.getElementById('btn-predict');
    const btnUpload = document.getElementById('btn-upload');
    const fileUpload = document.getElementById('file-upload');
    
    const btnFreehand = document.getElementById('btn-freehand');
    const btnLine = document.getElementById('btn-line');
    
    const placeholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const predChar = document.getElementById('pred-char');
    const predPhonetic = document.getElementById('pred-phonetic');
    const confidenceBars = document.getElementById('confidence-bars');

    // Drawing configuration
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 24; // Keep this thin for a good user experience
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    let currentTool = 'freehand';
    let canvasSnapshot = null;
    
    // Clear canvas to solid black
    function clearCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        saveState(); // Save initial empty state
    }
    
    // Drawing History (for Undo function)
    let history = [];
    const maxHistory = 15;
    
    function saveState() {
        if (history.length >= maxHistory) {
            history.shift();
        }
        history.push(canvas.toDataURL());
    }
    
    function undo() {
        if (history.length > 1) {
            history.pop(); // Remove current state
            const previousState = history[history.length - 1];
            const img = new Image();
            img.onload = () => {
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            };
            img.src = previousState;
        } else if (history.length === 1) {
            clearCanvas();
        }
    }

    clearCanvas(); // Run once at boot

    // Mouse and Touch drawing states
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let startX = 0;
    let startY = 0;

    function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        // Handle touch vs mouse
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * (canvas.width / rect.width),
            y: (clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        const coords = getCoords(e);
        lastX = coords.x;
        lastY = coords.y;
        startX = coords.x;
        startY = coords.y;
        
        canvasWrapper.classList.add('drawing');
        
        if (currentTool === 'line') {
            // Save snapshot before drawing preview line
            canvasSnapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
        } else {
            // Draw a single dot on click/tap for freehand
            ctx.beginPath();
            ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.fill();
        }
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault(); // Prevent scrolling on mobile touch
        
        const coords = getCoords(e);
        
        if (currentTool === 'line') {
            // Restore snapshot to clear previous preview line
            if (canvasSnapshot) {
                ctx.putImageData(canvasSnapshot, 0, 0);
            }
            // Draw straight line from start to current position
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(coords.x, coords.y);
            ctx.stroke();
        } else {
            // Freehand drawing
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(coords.x, coords.y);
            ctx.stroke();
            
            lastX = coords.x;
            lastY = coords.y;
        }
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            canvasWrapper.classList.remove('drawing');
            canvasSnapshot = null;
            saveState();
        }
    }

    // Tool switching events
    if (btnFreehand && btnLine) {
        btnFreehand.addEventListener('click', () => {
            currentTool = 'freehand';
            btnFreehand.classList.add('active');
            btnLine.classList.remove('active');
        });
        
        btnLine.addEventListener('click', () => {
            currentTool = 'line';
            btnLine.classList.add('active');
            btnFreehand.classList.remove('active');
        });
    }

    // Desktop Mouse Events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    // Mobile Touch Events
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    window.addEventListener('touchend', stopDrawing);

    // Button actions
    btnClear.addEventListener('click', () => {
        clearCanvas();
        placeholder.classList.remove('hidden');
        resultContent.classList.add('hidden');
    });
    
    btnUndo.addEventListener('click', undo);
    
    btnUpload.addEventListener('click', () => {
        fileUpload.click();
    });

    fileUpload.addEventListener('change', () => {
        const file = fileUpload.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                // Clear to solid white to default to light-background/notebook paper path
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Calculate fit scale (90% size)
                const scale = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.9;
                const w = img.width * scale;
                const h = img.height * scale;
                const x = (canvas.width - w) / 2;
                const y = (canvas.height - h) / 2;

                ctx.drawImage(img, x, y, w, h);
                saveState();

                // Auto-trigger prediction
                btnPredict.click();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });
    
    btnPredict.addEventListener('click', async () => {
        // Prepare base64
        const dataUrl = canvas.toDataURL('image/png');
        
        // Set loading UI state
        btnPredict.disabled = true;
        btnPredict.innerHTML = '<svg class="spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle><path d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z"></path></svg> Predicting...';

        // Add spinner rotation animation via JS-injected style if not in CSS
        if (!document.getElementById('spinner-style')) {
            const style = document.createElement('style');
            style.id = 'spinner-style';
            style.innerHTML = `
                .spinner { animation: rotate 1s linear infinite; }
                @keyframes rotate { 100% { transform: rotate(360deg); } }
            `;
            document.head.appendChild(style);
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            
            const result = await response.json();
            btnPredict.disabled = false;
            btnPredict.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> Predict';

            if (result.error) {
                alert("Error: " + result.error);
                return;
            }

            const predictions = result.predictions;
            if (predictions && predictions.length > 0) {
                // Populate primary prediction
                predChar.textContent = predictions[0].character;
                predPhonetic.textContent = predictions[0].phonetic;

                // Populate confidence bars
                confidenceBars.innerHTML = '';
                predictions.forEach(pred => {
                    const pct = Math.round(pred.confidence * 100);
                    const row = document.createElement('div');
                    row.className = 'confidence-row';
                    row.innerHTML = `
                        <div class="bar-labels">
                            <div>
                                <span class="bar-char">${pred.character}</span>
                                <span class="bar-name">(${pred.phonetic})</span>
                            </div>
                            <span class="bar-pct">${pct}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 0%"></div>
                        </div>
                    `;
                    confidenceBars.appendChild(row);
                    
                    // Trigger reflow then set width for smooth progress bar transition
                    setTimeout(() => {
                        row.querySelector('.bar-fill').style.width = `${pct}%`;
                    }, 50);
                });

                // Toggle views
                placeholder.classList.add('hidden');
                resultContent.classList.remove('hidden');
            }
        } catch (error) {
            btnPredict.disabled = false;
            btnPredict.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> Predict';
            console.error("Prediction failed:", error);
            alert("Network prediction request failed.");
        }
    });
});
