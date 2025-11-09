document.addEventListener('DOMContentLoaded', () => {

    // --- Get All Elements ---
    const navButtons = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');
    const captureBtn = document.getElementById('captureBtn');
    const fileInput = document.getElementById('fileInput');
    const videoFeed = document.getElementById('camera-feed');
    const canvas = document.getElementById('capture-canvas');
    const cameraContainer = document.getElementById('camera-container');
    const scannerOverlay = document.getElementById('scanner-overlay');
    const fallbackLabel = document.getElementById('file-fallback-label');
    
    // Result Page
    const resultSection = document.getElementById('result-section');
    const quickVerdict = document.getElementById('quickVerdict');
    const detailedReport = document.getElementById('detailedReport');
    const viewReportBtn = document.getElementById('viewReportBtn');
    const scanAgainBtn = document.getElementById('scanAgainBtn');

    // History Page
    const historyList = document.getElementById('historyList'); 
    
    // Chat Page
    const chatWindow = document.getElementById('chatWindow');
    const chatInput = document.getElementById('chatInput');
    const sendMsgBtn = document.getElementById('sendMsgBtn');
    
    // New UI Elements
    const globalLoader = document.getElementById('global-loader');
    const globalError = document.getElementById('global-error');
    const globalErrorText = document.querySelector('#global-error p');
    const closeErrorBtn = document.getElementById('close-error-btn');

    // --- Global State ---
    let cameraStream = null;
    window.latestReport = null;

    // --- NEW: UI Helper Functions ---
    function showLoader(show) {
        if (globalLoader) {
            globalLoader.style.display = show ? 'flex' : 'none';
        }
    }

    function showError(message) {
        if (globalError && globalErrorText) {
            globalErrorText.textContent = message;
            globalError.style.display = 'flex';
        } else {
            alert(message); // Fallback
        }
    }
    
    function hideError() {
        if (globalError) {
            globalError.style.display = 'none';
        }
    }
    
    if (closeErrorBtn) {
        closeErrorBtn.addEventListener('click', hideError);
    }
    
    // --- 1. Page Navigation Logic ---
    function activatePage(pageId) {
        hideError();
        pages.forEach(page => page.classList.remove('active'));
        navButtons.forEach(btn => btn.classList.remove('active'));

        const page = document.getElementById(pageId);
        if (page) page.classList.add('active');
        
        const activeNavBtn = document.querySelector(`.nav-btn[data-page="${pageId}"]`);
        if (activeNavBtn) activeNavBtn.classList.add('active');

        // Stop camera
        if (pageId !== 'page-scan' && cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            videoFeed.srcObject = null;
            cameraStream = null;
        } else if (pageId === 'page-scan' && !cameraStream) {
            startCamera();
        }

        if (pageId === 'page-history') {
            fetchHistory();
        }
    }

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            activatePage(button.dataset.page);
        });
    });

    if (document.getElementById('page-scan')) {
        activatePage('page-scan'); // Start on scan page if we're on index.html
    }

    // --- 2. Camera Logic ---
    async function startCamera() {
        if (cameraStream) return;
        
        hideError();
        if (cameraContainer) cameraContainer.style.display = 'block'; 
        if (scannerOverlay) scannerOverlay.style.display = 'block';
        if (videoFeed) videoFeed.style.display = 'block';
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            videoFeed.srcObject = stream;
            cameraStream = stream;
            if (fallbackLabel) fallbackLabel.style.display = 'none';
        } catch (err) {
            console.error("Camera error:", err);
            showError("Could not access camera. Please check permissions.");
            if (cameraContainer) cameraContainer.style.display = 'none'; 
            if (fallbackLabel) fallbackLabel.style.display = 'block';
            if (scannerOverlay) scannerOverlay.style.display = 'none';
            if (videoFeed) videoFeed.style.display = 'none';
        }
    }
    
    // --- 3. Capture & Upload Logic (FIXED) ---
    async function handleUpload(formData) {
        showLoader(true);
        hideError();
        if (resultSection) resultSection.style.display = 'none';
        
        // Hide camera UI
        if (scannerOverlay) scannerOverlay.style.display = 'none';
        if (videoFeed) videoFeed.style.display = 'none';
        if (cameraContainer) cameraContainer.style.display = 'none';
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        try {
            const res = await fetch("/analyze", { 
                method: "POST", 
                body: formData 
            });
            
            const data = await res.json();

            if (!res.ok) {
                // Show server-side error
                throw new Error(data.error || 'Analysis failed');
            }
            
            // Success! Show results.
            if (quickVerdict) quickVerdict.innerText = data.quick_verdict;
            window.latestReport = data.detailed_report;
            if (resultSection) resultSection.style.display = 'block'; 

        } catch (err) {
            console.error("Error during analysis:", err);
            showError(err.message);
            // On error, show camera again
            if (scanAgainBtn) scanAgainBtn.click();
        } finally {
            showLoader(false);
        }
    }

    // "Scan Again" button listener
    if (scanAgainBtn) {
        scanAgainBtn.addEventListener('click', () => {
            if (resultSection) resultSection.style.display = 'none';
            if (detailedReport) detailedReport.style.display = 'none'; // Hide report too
            startCamera(); 
        });
    }

    // Capture button
    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            const activePage = document.querySelector('.page.active');
            
            if (activePage && activePage.id === 'page-scan') {
                if (!cameraStream) {
                    showError("Camera not active. Please allow camera access.");
                    return;
                }
                canvas.width = videoFeed.videoWidth;
                canvas.height = videoFeed.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, 'scan.jpg');
                    const scanTypeRadio = document.querySelector('input[name="scan_type"]:checked');
                    const scanType = scanTypeRadio ? scanTypeRadio.value : 'label';
                    formData.append('scan_type', scanType);
                    
                    handleUpload(formData);
                }, 'image/jpeg', 0.9);
            
            } else {
                activatePage('page-scan');
            }
        });
    }

    // Fallback file input logic
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (!fileInput.files.length) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const scanTypeRadio = document.querySelector('input[name="scan_type"]:checked');
            const scanType = scanTypeRadio ? scanTypeRadio.value : 'label';
            formData.append('scan_type', scanType);
            handleUpload(formData);
        });
    }
    if (fallbackLabel) {
        fallbackLabel.addEventListener('click', () => fileInput.click());
    }

    // --- 4. Other Logic (FIXED) ---
    if (viewReportBtn) {
        viewReportBtn.addEventListener("click", () => {
            if (detailedReport) {
                detailedReport.style.display = detailedReport.style.display === "none" ? "block" : "none";
                if (detailedReport.style.display === "block" && window.latestReport) {
                    let html = "<table><tr><th>Detail</th><th>Value</th></tr>";
                    window.latestReport.forEach(r => {
                        html += `<tr><td>${r.nutrient}</td><td>${r.impact}</td></tr>`;
                    });
                    html += "</table>";
                    detailedReport.innerHTML = html;
                }
            }
        });
    }

    // Chatbot listener (FIXED)
    if (sendMsgBtn) {
        sendMsgBtn.addEventListener('click', () => {
            const msg = chatInput ? chatInput.value.trim() : '';
            if (msg) sendMessageToBot(msg);
        });
    }
    
    async function sendMessageToBot(msg) {
        if (!msg || !chatWindow) return;
        
        hideError();
        const userMsg = document.createElement("p");
        userMsg.textContent = "🧑: " + msg;
        chatWindow.appendChild(userMsg);
        if (chatInput) chatInput.value = "";
        chatWindow.scrollTop = chatWindow.scrollHeight;

        const botReply = document.createElement("p");
        botReply.textContent = "🤖: ...";
        chatWindow.appendChild(botReply);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || 'Chatbot failed');
            }
            
            botReply.textContent = "🤖: " + data.reply;
        } catch (err) {
            console.error('Chat error:', err);
            botReply.textContent = "🤖: Sorry, I'm having trouble right now. Please try again.";
        }
        
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
    
    // --- 5. History Page Logic ---

    // --- NEW: Function to start chat with a topic ---
    function startChatWithTopic(topic) {
        // 1. Switch to the chatbot page
        activatePage('page-chatbot');
        
        // 2. Programmatically "ask" about the topic
        const prefillMsg = `Tell me more about: "${topic}"`;
        sendMessageToBot(prefillMsg);
    }

    // --- NEW: Function to share a history item ---
    async function shareHistoryItem(filename, verdict) {
        const imageUrl = `${window.location.origin}/static/uploads/${filename}`;
        
        try {
            // 1. Fetch the image and convert it to a blob
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            
            // 2. Create a File object
            const file = new File([blob], filename, { type: blob.type });
            
            // 3. Create share data
            const shareData = {
                title: 'My NutriScanAI Result',
                text: `NutriScanAI verdict: ${verdict}`,
                files: [file]
            };

            // 4. Try to share
            if (navigator.canShare && navigator.canShare(shareData)) {
                await navigator.share(shareData);
            } else {
                alert("Sharing files is not supported on this browser.");
            }
        } catch (err) {
            console.error('Share failed:', err);
            alert("Sharing failed. You may need to be on a secure (https) connection or use a supported browser.");
        }
    }

    // --- NEW: Event Delegation for History List ---
    if (historyList) {
        historyList.addEventListener('click', (e) => {
            // Find the button that was clicked, even if the <i> icon was clicked
            const shareButton = e.target.closest('.history-share-btn');
            const chatButton = e.target.closest('.history-chat-btn');

            if (shareButton) {
                // Get the data we stored on the button
                const { filename, verdict } = shareButton.dataset;
                shareHistoryItem(filename, verdict);
                return; // Stop further checks
            }

            if (chatButton) {
                // Get the data from the button
                const { topic } = chatButton.dataset;
                startChatWithTopic(topic);
                return; // Stop further checks
            }
        });
    }
    
    // --- UPDATED: Helper function to create a history item (with buttons) ---
    function renderHistoryItem(scanData, prepend = false) {
        if (!historyList) return;
        
        const item = document.createElement('li');
        item.className = 'history-item';

        // Image
        const img = document.createElement('img');
        img.src = `static/uploads/${scanData.filename}`;
        img.alt = `Scan from ${scanData.timestamp}`;

        // Info (Verdict + Timestamp)
        const info = document.createElement('div');
        info.className = 'history-info';
        info.innerHTML = `<strong>${scanData.quick_verdict}</strong><p>${scanData.timestamp}</p>`;

        // --- NEW: Action Buttons Container ---
        const actions = document.createElement('div');
        actions.className = 'history-item-actions';

        // Chat Button
        const chatBtn = document.createElement('button');
        chatBtn.className = 'history-chat-btn';
        chatBtn.innerHTML = '<i class="fa-solid fa-robot"></i>';
        // Store data on the button itself for the listener
        chatBtn.dataset.topic = scanData.ocr_text || scanData.quick_verdict; 
        
        // Share Button
        const shareBtn = document.createElement('button');
        shareBtn.className = 'history-share-btn';
        shareBtn.innerHTML = '<i class="fa-solid fa-share"></i>';
        // Store data on the button
        shareBtn.dataset.filename = scanData.filename;
        shareBtn.dataset.verdict = scanData.quick_verdict;
        
        actions.appendChild(chatBtn);
        actions.appendChild(shareBtn);
        
        // Assemble the item
        item.appendChild(img);
        item.appendChild(info);
        item.appendChild(actions); // Add the new actions div

        if (prepend) {
            historyList.prepend(item); 
        } else {
            historyList.appendChild(item); 
        }
    }
  
    // Function to Fetch and Render History (updated)
    async function fetchHistory() {
        if (!historyList) return;
        
        historyList.innerHTML = '<p class="loading-msg">Loading history...</p>';
        try {
            const res = await fetch('/history'); // Assumes you're logged in
            if (!res.ok) {
                 // If we get a 401 (unauthorized), redirect to login
                 if (res.status === 401) {
                    window.location.href = '/login';
                 }
                 throw new Error('Failed to fetch history');
            }
            
            const data = await res.json(); 
            historyList.innerHTML = ''; 

            if (data.length === 0) {
                historyList.innerHTML = '<p class="loading-msg">No scan history found.</p>';
                return;
            }
            
            data.forEach(scanData => {
                renderHistoryItem(scanData, false); 
            });

        } catch (err) {
            console.error('Error loading history:', err);
            historyList.innerHTML = '<p class="loading-msg">Error loading history. Please try again.</p>';
        }
    }

});