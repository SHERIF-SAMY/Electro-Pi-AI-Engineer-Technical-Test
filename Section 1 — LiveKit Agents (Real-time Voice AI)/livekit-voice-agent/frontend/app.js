document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    const muteBtn = document.getElementById('mute-btn');
    const roomNameInput = document.getElementById('room-name');
    const userNameInput = document.getElementById('user-name');
    
    const statusText = document.getElementById('connection-status');
    const statusDot = document.getElementById('status-dot');
    const agentStateText = document.getElementById('agent-state-text');
    const agentSubtitle = document.getElementById('agent-subtitle');
    const agentGlow = document.getElementById('agent-glow');
    const audioContainer = document.getElementById('audio-container');
    const logsContainer = document.getElementById('logs-container');
    const clearLogsBtn = document.getElementById('clear-logs-btn');
    
    // LiveKit Room instance
    let room = null;
    let isMuted = false;

    // Connect to LiveKit Room
    async function connectToRoom() {
        const roomName = roomNameInput.value.trim() || 'support-room-1';
        const participantName = userNameInput.value.trim() || 'Customer';

        try {
            // UI Update: Connecting
            setUIConnecting();

            // 1. Fetch token from backend
            const response = await fetch('/api/token', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ room_name: roomName, participant_name: participantName })
            });

            if (!response.ok) {
                throw new Error('Failed to get token');
            }

            const data = await response.json();
            const token = data.token;
            
            // Note: Update this URL if your LiveKit server is hosted elsewhere
            const wsUrl = 'ws://localhost:7880';

            // 2. Initialize Room
            room = new LivekitClient.Room({
                adaptiveStream: true,
                dynacast: true,
            });

            // 3. Set up event listeners
            room
                .on(LivekitClient.RoomEvent.Connected, () => {
                    console.log('Connected to room!');
                    setUIConnected();
                })
                .on(LivekitClient.RoomEvent.Disconnected, () => {
                    console.log('Disconnected from room');
                    setUIDisconnected();
                })
                .on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
                    console.log('Track subscribed:', track.kind, 'from', participant.identity);
                    if (track.kind === 'audio') {
                        // The agent is speaking, show visualizer/glow
                        const element = track.attach();
                        audioContainer.appendChild(element);
                        agentGlow.classList.add('active');
                        agentStateText.innerText = "Zara is speaking...";
                        
                        // Stop glow when audio stops
                        // In a real app you'd use track.on(TrackEvent.AudioPlaybackStarted / Stopped)
                        // Or watch AudioContext analyser.
                        element.onended = () => {
                             agentGlow.classList.remove('active');
                             agentStateText.innerText = "Listening...";
                        };
                    }
                })
                .on(LivekitClient.RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
                    if (track.kind === 'audio') {
                        track.detach();
                        agentGlow.classList.remove('active');
                    }
                })
                .on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
                    if (participant.identity.toLowerCase().includes('agent')) {
                        agentStateText.innerText = "Zara has joined!";
                        appendLogToUI({type: 'system', text: 'Agent joined the room'});
                    }
                })
                .on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind, topic) => {
                    if (topic === "pipeline_logs") {
                        try {
                            const strData = new TextDecoder().decode(payload);
                            const logData = JSON.parse(strData);
                            appendLogToUI(logData);
                        } catch (e) {
                            console.error("Error parsing pipeline log", e);
                        }
                    }
                });

            // 4. Connect to the room
            await room.connect(wsUrl, token);
            
            // 5. Publish Microphone
            await room.localParticipant.setMicrophoneEnabled(true);
            
        } catch (error) {
            console.error('Connection error:', error);
            alert(`Could not connect: ${error.message}`);
            setUIDisconnected();
        }
    }

    // Disconnect from room
    async function disconnectFromRoom() {
        if (room) {
            await room.disconnect();
            room = null;
        }
        setUIDisconnected();
    }

    // Toggle Microphone
    async function toggleMute() {
        if (!room) return;
        
        isMuted = !isMuted;
        await room.localParticipant.setMicrophoneEnabled(!isMuted);
        
        if (isMuted) {
            muteBtn.classList.add('muted');
            // Change to strikethrough mic SVG
            muteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`;
        } else {
            muteBtn.classList.remove('muted');
            // Change to normal mic SVG
            muteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`;
        }
    }

    // UI State Helpers
    function setUIConnecting() {
        connectBtn.disabled = true;
        connectBtn.innerText = 'Connecting...';
        statusText.innerText = 'Connecting...';
        statusText.className = 'status-text';
        statusDot.className = 'status-dot';
    }

    function setUIConnected() {
        connectBtn.disabled = true;
        connectBtn.innerText = 'Connected';
        disconnectBtn.disabled = false;
        muteBtn.disabled = false;
        
        statusText.innerText = 'Connected';
        statusText.className = 'status-text connected';
        statusDot.className = 'status-dot connected';
        
        agentStateText.innerText = 'Listening...';
        agentSubtitle.innerText = 'Speak into your microphone';
        
        roomNameInput.disabled = true;
        userNameInput.disabled = true;
    }

    function setUIDisconnected() {
        connectBtn.disabled = false;
        connectBtn.innerText = 'Connect & Talk';
        disconnectBtn.disabled = true;
        muteBtn.disabled = true;
        isMuted = false;
        muteBtn.classList.remove('muted');
        muteBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`;
        
        statusText.innerText = 'Disconnected';
        statusText.className = 'status-text disconnected';
        statusDot.className = 'status-dot disconnected';
        
        agentStateText.innerText = 'Ready to connect';
        agentSubtitle.innerText = 'Connect to start talking with Zara.';
        agentGlow.classList.remove('active');
        audioContainer.innerHTML = '';
        
        roomNameInput.disabled = false;
        userNameInput.disabled = false;
    }

    function appendLogToUI(logData) {
        if (!logsContainer) return;
        const entry = document.createElement('div');
        entry.className = `log-entry ${logData.type}`;
        
        let prefix = "";
        if (logData.type === 'user') prefix = "👤 You: ";
        else if (logData.type === 'agent') prefix = "🤖 Zara: ";
        else if (logData.type === 'tool') prefix = "⚡ ";
        
        entry.innerText = prefix + logData.text;
        logsContainer.appendChild(entry);
        
        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', () => {
            if (logsContainer) {
                logsContainer.innerHTML = '<div class="log-entry system">Logs cleared</div>';
            }
        });
    }

    // Event Listeners
    connectBtn.addEventListener('click', connectToRoom);
    disconnectBtn.addEventListener('click', disconnectFromRoom);
    muteBtn.addEventListener('click', toggleMute);
});
