/**
 * FACE 2.0 - Mobile Web Biometric Suite
 * Pure Vanilla JavaScript Client
 *
 * Features:
 * - WebRTC camera streaming with rear/front camera toggle
 * - Low-latency WebSocket video pipeline to /ws/recognize
 * - Strict client-side backpressure lock preventing Wi-Fi queue buildup
 * - Precision canvas overlay renderer matching desktop OpenCV aesthetics
 * - Slide-up recognized identity bottom sheet
 * - Multi-angle Camera Shutter enrollment studio with live guide
 * - Multi-image Gallery Upload enrollment studio
 * - Enrolled biometric directory with live search and delete confirmation
 * - Responsive mobile toast notifications
 */

(function () {
  'use strict';

  // =========================================================================
  // APPLICATION STATE
  // =========================================================================
  const state = {
    // Current Active Tab
    currentTab: 'tab-live', // 'tab-live' | 'tab-enroll' | 'tab-directory'
    enrollMode: 'shutter',  // 'shutter' | 'gallery'

    // Camera & MediaStream
    cameraStream: null,
    facingMode: 'environment', // Defaults to rear camera per user specification
    isStreamPaused: false,

    // WebSocket & Live Inference Loop
    ws: null,
    wsConnected: false,
    wsReconnectTimeout: null,
    wsReconnectAttempts: 0,
    isAwaitingResponse: false, // Strict backpressure control
    backpressureWatchdog: null, // Watchdog timer preventing permanent backpressure stall
    lastFrameTime: performance.now(),
    fps: 0,
    fpsIntervalStart: performance.now(),
    fpsFrameCounter: 0,

    // Auto-hide timer for recognition bottom sheet
    recogSheetTimer: null,

    // Enrollment Studio: Camera Shutter
    shutterPhotos: [], // Array of base64 data URLs: ["data:image/jpeg;base64,..."]
    maxShutterPhotos: 10,

    // Enrollment Studio: Gallery Upload
    galleryFiles: [], // Array of File objects

    // Directory
    directoryMembers: [],
    memberToDelete: null,
  };

  // =========================================================================
  // DOM ELEMENT SELECTORS
  // =========================================================================
  const el = {
    // Tabs & Navigation
    navTabs: document.querySelectorAll('.nav-tab'),
    tabScreens: document.querySelectorAll('.tab-screen'),

    // Top Floating Bar (Live Tab)
    streamStatusDot: document.getElementById('streamStatusDot'),
    streamStatusText: document.getElementById('streamStatusText'),
    streamFps: document.getElementById('streamFps'),
    btnFlipCamera: document.getElementById('btnFlipCamera'),
    btnToggleStream: document.getElementById('btnToggleStream'),
    iconStreamPause: document.getElementById('iconStreamPause'),
    iconStreamPlay: document.getElementById('iconStreamPlay'),

    // Live Video & Canvas
    cameraVideo: document.getElementById('cameraVideo'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    cameraPlaceholder: document.getElementById('cameraPlaceholder'),
    placeholderText: document.getElementById('placeholderText'),
    hiddenCanvas: document.getElementById('hiddenFrameCanvas'),

    // Slide-up Bottom Card (Live Tab)
    recognizedCard: document.getElementById('recognizedCard'),
    recogAvatar: document.getElementById('recogAvatar'),
    recogInitials: document.getElementById('recogInitials'),
    recogName: document.getElementById('recogName'),
    recogRole: document.getElementById('recogRole'),
    recogScore: document.getElementById('recogScore'),
    recogConfidenceValue: document.getElementById('recogConfidenceValue'),
    recogScoreBar: document.getElementById('recogScoreBar'),
    recogNotes: document.getElementById('recogNotes'),
    recogStatus: document.getElementById('recogStatus'),
    btnCloseCard: document.getElementById('btnCloseCard'),

    // Enrollment Tabs / Segments
    segmentShutter: document.getElementById('segment-shutter'),
    segmentGallery: document.getElementById('segment-gallery'),
    enrollShutterView: document.getElementById('enroll-shutter-view'),
    enrollGalleryView: document.getElementById('enroll-gallery-view'),

    // Enrollment: Shutter View
    enrollVideo: document.getElementById('enrollVideo'),
    btnEnrollFlip: document.getElementById('btnEnrollFlip'),
    viewfinderFlash: document.getElementById('viewfinderFlash'),
    btnShutter: document.getElementById('btnShutter'),
    shutterCountBadge: document.getElementById('shutterCountBadge'),
    btnClearShutterPhotos: document.getElementById('btnClearShutterPhotos'),
    shutterThumbnails: document.getElementById('shutterThumbnails'),
    thumbnailPlaceholder: document.getElementById('thumbnailPlaceholder'),
    thumbnailCountLabel: document.getElementById('thumbnailCountLabel'),
    shutterEnrollForm: document.getElementById('shutterEnrollForm'),
    shutterName: document.getElementById('shutterName'),
    shutterRole: document.getElementById('shutterRole'),
    shutterNotes: document.getElementById('shutterNotes'),
    btnSaveEnrollCaptures: document.getElementById('btnSaveEnrollCaptures'),

    // Enrollment: Gallery View
    galleryDropzone: document.getElementById('galleryDropzone'),
    galleryFileInput: document.getElementById('galleryFileInput'),
    btnSelectFiles: document.getElementById('btnSelectFiles'),
    galleryPreviewGrid: document.getElementById('galleryPreviewGrid'),
    galleryCountLabel: document.getElementById('galleryCountLabel'),
    galleryEnrollForm: document.getElementById('galleryEnrollForm'),
    galleryName: document.getElementById('galleryName'),
    galleryRole: document.getElementById('galleryRole'),
    galleryNotes: document.getElementById('galleryNotes'),
    btnUploadEnrollGallery: document.getElementById('btnUploadEnrollGallery'),

    // Directory View
    directorySearch: document.getElementById('directorySearch'),
    btnClearSearch: document.getElementById('btnClearSearch'),
    btnRefreshDirectory: document.getElementById('btnRefreshDirectory'),
    directoryCountBadge: document.getElementById('directoryCountBadge'),
    directoryFilterHint: document.getElementById('directoryFilterHint'),
    directoryList: document.getElementById('directoryList'),
    directoryEmptyState: document.getElementById('directoryEmptyState'),
    emptyTitle: document.getElementById('emptyTitle'),
    emptyDesc: document.getElementById('emptyDesc'),

    // Toast & Modal
    toastContainer: document.getElementById('toastContainer'),
    confirmModal: document.getElementById('confirmModal'),
    deleteTargetName: document.getElementById('deleteTargetName'),
    btnCancelDelete: document.getElementById('btnCancelDelete'),
    btnConfirmDelete: document.getElementById('btnConfirmDelete'),
  };

  const hiddenCtx = el.hiddenCanvas.getContext('2d', { willReadFrequently: true });
  const overlayCtx = el.overlayCanvas.getContext('2d');

  // =========================================================================
  // NOTIFICATIONS (TOASTS)
  // =========================================================================
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconSvg = '';
    if (type === 'success') {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (type === 'error') {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
    } else {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }

    toast.innerHTML = `<span>${iconSvg}</span><span>${message}</span>`;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-out');
      toast.addEventListener('animationend', () => toast.remove());
    }, duration);
  }

  // =========================================================================
  // CAMERA WEBRTC STREAMING
  // =========================================================================
  async function initCamera(facingMode = 'environment') {
    state.facingMode = facingMode;

    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }

    setPlaceholder(true, 'Accessing camera...');

    const constraints = {
      video: {
        facingMode: { ideal: state.facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.cameraStream = stream;

      // Bind stream to both Live tab video and Enrollment viewfinder video
      el.cameraVideo.srcObject = stream;
      el.enrollVideo.srcObject = stream;

      const waitForMetadata = (vid) =>
        new Promise((resolve) => {
          if (vid.readyState >= 1) {
            resolve();
            return;
          }
          let resolved = false;
          const onLoaded = () => {
            if (!resolved) {
              resolved = true;
              resolve();
            }
          };
          vid.addEventListener('loadedmetadata', onLoaded, { once: true });
          // Fallback timer: prevents Promise.all hang if browser delays hidden element metadata
          setTimeout(onLoaded, 1200);
        });

      await Promise.all([
        waitForMetadata(el.cameraVideo),
        waitForMetadata(el.enrollVideo),
      ]);

      await el.cameraVideo.play().catch((err) => console.warn('Camera video play error:', err));
      await el.enrollVideo.play().catch((err) => console.warn('Enroll video play error:', err));

      // Configure overlay canvas resolution to match video stream
      updateCanvasDimensions();
      setPlaceholder(false);
      showToast(`Camera active (${state.facingMode === 'environment' ? 'Rear' : 'Front'})`, 'info', 2000);
    } catch (err) {
      console.error('Camera initialization failed:', err);
      setPlaceholder(true, `Camera error: ${err.name || err.message}`);
      showToast(`Could not open camera: ${err.message}`, 'error', 4500);
    }
  }

  function toggleCameraFacing() {
    const nextMode = state.facingMode === 'environment' ? 'user' : 'environment';
    initCamera(nextMode);
  }

  function setPlaceholder(visible, message = '') {
    if (visible) {
      el.cameraPlaceholder.classList.remove('hidden');
      if (message) el.placeholderText.textContent = message;
    } else {
      el.cameraPlaceholder.classList.add('hidden');
    }
  }

  function updateCanvasDimensions() {
    const vw = el.cameraVideo.videoWidth || 640;
    const vh = el.cameraVideo.videoHeight || 480;
    el.overlayCanvas.width = vw;
    el.overlayCanvas.height = vh;
    el.hiddenCanvas.width = vw;
    el.hiddenCanvas.height = vh;
  }

  window.addEventListener('resize', () => {
    updateCanvasDimensions();
  });

  // =========================================================================
  // WEBSOCKET PIPELINE (/ws/recognize) & BACKPRESSURE CONTROL
  // =========================================================================
  function getWsUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/recognize`;
  }

  function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = getWsUrl();
    updateConnectionStatus('connecting');

    try {
      state.ws = new WebSocket(wsUrl);
      state.ws.binaryType = 'blob'; // JPEG frames transmitted as binary Blobs

      state.ws.onopen = () => {
        console.log('[WS] Connected to recognition server:', wsUrl);
        state.wsConnected = true;
        state.wsReconnectAttempts = 0;
        state.isAwaitingResponse = false;
        updateConnectionStatus('connected');
        showToast('Recognition server connected', 'success', 2000);
      };

      state.ws.onmessage = (event) => {
        // Backend has acknowledged the frame and responded: release backpressure lock!
        clearTimeout(state.backpressureWatchdog);
        state.isAwaitingResponse = false;
        handleRecognitionResponse(event.data);
      };

      state.ws.onerror = (err) => {
        console.warn('[WS] Socket error:', err);
        clearTimeout(state.backpressureWatchdog);
        state.isAwaitingResponse = false;
      };

      state.ws.onclose = () => {
        console.log('[WS] Connection closed');
        clearTimeout(state.backpressureWatchdog);
        state.wsConnected = false;
        state.isAwaitingResponse = false;
        updateConnectionStatus('disconnected');

        // Exponential backoff reconnect
        const delay = Math.min(1000 * Math.pow(1.5, state.wsReconnectAttempts), 10000);
        state.wsReconnectAttempts++;
        clearTimeout(state.wsReconnectTimeout);
        state.wsReconnectTimeout = setTimeout(connectWebSocket, delay);
      };
    } catch (e) {
      console.error('[WS] Connection setup error:', e);
      updateConnectionStatus('disconnected');
    }
  }

  function updateConnectionStatus(status) {
    el.streamStatusDot.className = 'status-dot';
    if (status === 'connected') {
      el.streamStatusDot.classList.add('connected');
      el.streamStatusText.textContent = state.isStreamPaused ? 'Paused' : 'Streaming';
    } else if (status === 'connecting') {
      el.streamStatusText.textContent = 'Connecting...';
    } else {
      el.streamStatusDot.classList.add('disconnected');
      el.streamStatusText.textContent = 'Offline';
      el.streamFps.textContent = '0.0 FPS';
    }
  }

  // =========================================================================
  // LIVE FRAME CAPTURE & STREAMING LOOP (WITH BACKPRESSURE)
  // =========================================================================
  function scheduleStreamLoop() {
    requestAnimationFrame(streamLoop);
  }

  function streamLoop() {
    // Only capture and send frames when:
    // 1. Live tab is currently active
    // 2. Stream is not paused by user
    // 3. WebSocket is open and connected
    // 4. Backpressure lock is FALSE (we are not waiting for previous frame response)
    // 5. Camera video is playing and ready
    const canSend =
      state.currentTab === 'tab-live' &&
      !state.isStreamPaused &&
      state.wsConnected &&
      state.ws &&
      state.ws.readyState === WebSocket.OPEN &&
      !state.isAwaitingResponse &&
      el.cameraVideo.readyState >= 2 &&
      !el.cameraVideo.paused;

    if (canSend) {
      captureAndSendFrame();
    }

    scheduleStreamLoop();
  }

  function captureAndSendFrame() {
    const vw = el.cameraVideo.videoWidth;
    const vh = el.cameraVideo.videoHeight;
    if (!vw || !vh) return;

    if (el.hiddenCanvas.width !== vw || el.hiddenCanvas.height !== vh) {
      el.hiddenCanvas.width = vw;
      el.hiddenCanvas.height = vh;
    }

    // Draw current video frame to offscreen canvas
    hiddenCtx.drawImage(el.cameraVideo, 0, 0, vw, vh);

    // Set backpressure flag: wait for backend acknowledgement before sending next frame
    state.isAwaitingResponse = true;
    clearTimeout(state.backpressureWatchdog);
    state.backpressureWatchdog = setTimeout(() => {
      if (state.isAwaitingResponse) {
        console.warn('[WS] Frame response timeout (3.5s), releasing backpressure lock.');
        state.isAwaitingResponse = false;
      }
    }, 3500);

    // Convert frame to JPEG Blob and dispatch over WebSocket
    el.hiddenCanvas.toBlob(
      (blob) => {
        if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
          clearTimeout(state.backpressureWatchdog);
          state.isAwaitingResponse = false;
          return;
        }

        try {
          state.ws.send(blob);
          // Calculate client-side frame rate
          trackFps();
        } catch (err) {
          console.error('[WS] Send error:', err);
          clearTimeout(state.backpressureWatchdog);
          state.isAwaitingResponse = false;
        }
      },
      'image/jpeg',
      0.72 // High quality, low bandwidth compression for Wi-Fi
    );
  }

  function trackFps() {
    state.fpsFrameCounter++;
    const now = performance.now();
    const elapsed = now - state.fpsIntervalStart;

    if (elapsed >= 1000) {
      state.fps = (state.fpsFrameCounter / (elapsed / 1000)).toFixed(1);
      el.streamFps.textContent = `${state.fps} FPS`;
      state.fpsFrameCounter = 0;
      state.fpsIntervalStart = now;
    }
  }

  // =========================================================================
  // CANVAS OVERLAY RENDERER (DESKTOP OPENCV AESTHETICS)
  // =========================================================================
  function handleRecognitionResponse(rawPayload) {
    let data;
    try {
      data = typeof rawPayload === 'string' ? JSON.parse(rawPayload) : JSON.parse(new TextDecoder().decode(rawPayload));
    } catch (e) {
      console.warn('[WS] Malformed JSON response:', e);
      return;
    }

    // Backend FPS indicator sync (if provided)
    if (data.fps !== undefined && Number(data.fps) > 0) {
      el.streamFps.textContent = `${Number(data.fps).toFixed(1)} FPS`;
    }

    // Clear overlay canvas
    overlayCtx.clearRect(0, 0, el.overlayCanvas.width, el.overlayCanvas.height);

    const detections = data.faces || data.detections || data.results || [];
    let topKnownFace = null;

    if (detections.length === 0) {
      // Clear overlay and start auto-dismiss timer for bottom card
      scheduleBottomCardDismiss(2500);
      return;
    }

    const srcW = data.frame_width || data.width || el.overlayCanvas.width;
    const srcH = data.frame_height || data.height || el.overlayCanvas.height;
    const scaleX = el.overlayCanvas.width / (srcW || 1);
    const scaleY = el.overlayCanvas.height / (srcH || 1);

    // Draw all detected faces
    for (const item of detections) {
      const face = normalizeFaceData(item, scaleX, scaleY);
      drawFaceBoundingBox(face);

      // Track highest confidence known face for bottom sheet
      if (face.isKnown) {
        if (!topKnownFace || face.score > topKnownFace.score) {
          topKnownFace = face;
        }
      }
    }

    if (topKnownFace) {
      showRecognizedCard(topKnownFace);
    } else {
      scheduleBottomCardDismiss(2000);
    }
  }

  /**
   * Normalizes heterogeneous face detection payload schemas.
   */
  function normalizeFaceData(item, scaleX = 1, scaleY = 1) {
    let left = 0, top = 0, right = 0, bottom = 0;
    const box = item.box || item.bbox;

    if (Array.isArray(box)) {
      [left, top, right, bottom] = box;
    } else if (box && typeof box === 'object') {
      if ('left' in box) {
        left = box.left;
        top = box.top;
        right = box.right;
        bottom = box.bottom;
      } else if ('x' in box) {
        left = box.x;
        top = box.y;
        right = box.x + box.width;
        bottom = box.y + box.height;
      }
    }

    left *= scaleX;
    right *= scaleX;
    top *= scaleY;
    bottom *= scaleY;

    const name = item.name || 'Unknown';
    const isKnown = item.is_known !== undefined ? !!item.is_known : name.toLowerCase() !== 'unknown';
    let score = item.score !== undefined ? item.score : item.confidence || 0.0;
    if (score > 1.0) score = score / 100.0;

    const details = item.details || item.info || {};

    return {
      left: Math.round(left),
      top: Math.round(top),
      right: Math.round(right),
      bottom: Math.round(bottom),
      name,
      isKnown,
      score,
      details,
    };
  }

  /**
   * Render bounding box and solid label badge matching overlay.py styling:
   * - Emerald Green (COLOR_KNOWN = rgb(0, 200, 0)) for known identities
   * - Crimson Red (COLOR_UNKNOWN = rgb(230, 60, 0)) for unknown faces
   * - Solid background rectangle for guaranteed readability
   */
  function drawFaceBoundingBox(face) {
    const { left, top, right, bottom, name, isKnown, score, details } = face;
    const width = right - left;
    const height = bottom - top;
    if (width <= 0 || height <= 0) return;

    // Palette Colors matching overlay.py
    const colorPrimary = isKnown ? '#00c800' : '#e63c00'; // Emerald Green vs Crimson
    const textColor = '#ffffff';

    overlayCtx.save();

    // 1. Draw Face Bounding Box
    overlayCtx.strokeStyle = colorPrimary;
    overlayCtx.lineWidth = 3;
    overlayCtx.lineJoin = 'round';
    overlayCtx.strokeRect(left, top, width, height);

    // 2. Format Badge Text: "Name (Score) | Role"
    const scoreStr = (score * 100).toFixed(0) + '%';
    let label = isKnown ? `${name} (${scoreStr})` : `Unknown (${scoreStr})`;

    let roleText = '';
    if (isKnown && details) {
      if (typeof details === 'object') {
        roleText = details.role || details.department || '';
      } else if (typeof details === 'string') {
        roleText = details;
      }
    }
    if (roleText) {
      label += ` | ${roleText}`;
    }

    // 3. Compute Text Badge Dimensions
    const fontSize = Math.max(13, Math.min(20, Math.round(width * 0.08)));
    overlayCtx.font = `bold ${fontSize}px sans-serif`;
    const textMetrics = overlayCtx.measureText(label);
    const textWidth = textMetrics.width;
    const paddingX = 8;
    const paddingY = 5;
    const badgeHeight = fontSize + paddingY * 2;
    const badgeWidth = textWidth + paddingX * 2;

    // Place badge above bbox if room permits, otherwise below
    let badgeX = left;
    let badgeY = top - badgeHeight;
    if (badgeY < 0) {
      badgeY = bottom;
    }

    // Keep badge within horizontal canvas boundaries
    if (badgeX + badgeWidth > el.overlayCanvas.width) {
      badgeX = Math.max(0, el.overlayCanvas.width - badgeWidth - 4);
    }

    // 4. Draw SOLID Background Badge (High Readability)
    overlayCtx.fillStyle = colorPrimary;
    overlayCtx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);

    // 5. Render Label Text
    overlayCtx.fillStyle = textColor;
    overlayCtx.textBaseline = 'top';
    overlayCtx.fillText(label, badgeX + paddingX, badgeY + paddingY);

    overlayCtx.restore();
  }

  // =========================================================================
  // SLIDE-UP RECOGNIZED PERSON CARD
  // =========================================================================
  function showRecognizedCard(face) {
    clearTimeout(state.recogSheetTimer);

    el.recogName.textContent = face.name;
    const initials = face.name
      .split(' ')
      .map((w) => w[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
    el.recogInitials.textContent = initials || 'ID';

    const matchPercent = Math.round(face.score * 100);
    el.recogScore.textContent = `${matchPercent}% Match`;
    el.recogConfidenceValue.textContent = face.score.toFixed(2);
    el.recogScoreBar.style.width = `${Math.min(matchPercent, 100)}%`;

    // Role & Details
    let role = 'Enrolled Member';
    let notes = 'Biometric template verified in local SQLite database.';
    if (face.details && typeof face.details === 'object') {
      if (face.details.role) role = face.details.role;
      if (face.details.department) role = `${role} • ${face.details.department}`;
      if (face.details.notes) notes = face.details.notes;
    } else if (typeof face.details === 'string' && face.details.trim()) {
      role = face.details;
    }

    el.recogRole.textContent = role;
    el.recogNotes.textContent = notes;
    el.recogStatus.textContent = 'Verified Identity';
    el.recogStatus.className = 'detail-value text-emerald';

    el.recognizedCard.classList.add('visible');
  }

  function hideRecognizedCard() {
    el.recognizedCard.classList.remove('visible');
    clearTimeout(state.recogSheetTimer);
  }

  function scheduleBottomCardDismiss(delayMs = 3000) {
    clearTimeout(state.recogSheetTimer);
    state.recogSheetTimer = setTimeout(() => {
      hideRecognizedCard();
    }, delayMs);
  }

  // =========================================================================
  // ENROLLMENT STUDIO - SUB-VIEW A: CAMERA SHUTTER
  // =========================================================================
  function handleSnapShutter() {
    const video = el.enrollVideo.readyState >= 2 ? el.enrollVideo : el.cameraVideo;
    if (!video || !video.videoWidth) {
      showToast('Camera feed not ready yet', 'error');
      return;
    }

    // Flash effect
    el.viewfinderFlash.classList.add('flash');
    setTimeout(() => el.viewfinderFlash.classList.remove('flash'), 120);

    // Haptic vibration feedback if supported
    if ('vibrate' in navigator) {
      try { navigator.vibrate(35); } catch (_) {}
    }

    // Capture frame to temporary canvas
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = video.videoWidth;
    snapCanvas.height = video.videoHeight;
    const snapCtx = snapCanvas.getContext('2d');
    snapCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

    const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.88);
    state.shutterPhotos.push(dataUrl);

    renderShutterThumbnails();
    showToast(`Snapped photo ${state.shutterPhotos.length}`, 'info', 1200);
  }

  function renderShutterThumbnails() {
    const count = state.shutterPhotos.length;
    el.thumbnailCountLabel.textContent = `${count} captured`;
    el.shutterCountBadge.textContent = `Photos: ${count} / 5 (${count >= 3 ? 'Ready' : 'Min 3'})`;

    if (count >= 3) {
      el.shutterCountBadge.classList.add('ready');
    } else {
      el.shutterCountBadge.classList.remove('ready');
    }

    if (count > 0) {
      el.btnClearShutterPhotos.classList.remove('hidden');
      el.thumbnailPlaceholder.classList.add('hidden');
    } else {
      el.btnClearShutterPhotos.classList.add('hidden');
      el.thumbnailPlaceholder.classList.remove('hidden');
    }

    // Rebuild thumbnails container
    el.shutterThumbnails.querySelectorAll('.thumb-card').forEach((el) => el.remove());

    state.shutterPhotos.forEach((dataUrl, idx) => {
      const card = document.createElement('div');
      card.className = 'thumb-card';
      card.innerHTML = `
        <img src="${dataUrl}" alt="Capture ${idx + 1}" loading="lazy">
        <span class="thumb-index-tag">#${idx + 1}</span>
        <button type="button" class="thumb-delete-btn" data-index="${idx}" aria-label="Delete photo">&times;</button>
      `;

      card.querySelector('.thumb-delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteShutterPhoto(idx);
      });

      el.shutterThumbnails.appendChild(card);
    });

    // Auto-scroll to end
    el.shutterThumbnails.scrollLeft = el.shutterThumbnails.scrollWidth;
  }

  function deleteShutterPhoto(index) {
    state.shutterPhotos.splice(index, 1);
    renderShutterThumbnails();
  }

  function clearAllShutterPhotos() {
    state.shutterPhotos = [];
    renderShutterThumbnails();
    showToast('Cleared all snapped photos', 'info', 1500);
  }

  async function handleShutterEnrollSubmit(e) {
    e.preventDefault();

    const name = el.shutterName.value.trim();
    const role = el.shutterRole.value.trim();
    const notes = el.shutterNotes.value.trim();

    if (!name) {
      showToast('Please enter member name', 'error');
      el.shutterName.focus();
      return;
    }

    if (state.shutterPhotos.length === 0) {
      showToast('Snap at least 1 photo (3 recommended) to enroll', 'error');
      return;
    }

    setButtonLoading(el.btnSaveEnrollCaptures, true, 'Enrolling...');

    const payload = {
      name,
      role,
      notes,
      images: state.shutterPhotos,
    };

    try {
      const response = await fetch('/api/enroll/captures', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(result.detail || result.error || result.message || `HTTP ${response.status}`);
      }

      showToast(`Successfully enrolled '${name}'!`, 'success', 3500);

      // Reset form & captures
      el.shutterEnrollForm.reset();
      clearAllShutterPhotos();

      // Switch to directory to see new enrollment
      switchTab('tab-directory');
      loadDirectory();
    } catch (err) {
      console.error('Enrollment error:', err);
      showToast(`Enrollment failed: ${err.message}`, 'error', 4500);
    } finally {
      setButtonLoading(el.btnSaveEnrollCaptures, false, 'Save & Enroll Identity');
    }
  }

  // =========================================================================
  // ENROLLMENT STUDIO - SUB-VIEW B: GALLERY UPLOAD
  // =========================================================================
  function handleGalleryFilesSelected(fileList) {
    const files = Array.from(fileList).filter((f) => f.type.startsWith('image/'));
    if (files.length === 0) {
      showToast('Please select valid image files', 'error');
      return;
    }

    state.galleryFiles = state.galleryFiles.concat(files);
    renderGalleryPreviews();
    showToast(`Added ${files.length} photo(s)`, 'info', 1500);
  }

  function renderGalleryPreviews() {
    const count = state.galleryFiles.length;
    el.galleryCountLabel.textContent = `${count} selected`;

    el.galleryPreviewGrid.innerHTML = '';

    if (count === 0) {
      el.galleryPreviewGrid.innerHTML = '<div class="gallery-empty-msg">No images selected yet.</div>';
      return;
    }

    state.galleryFiles.forEach((file, idx) => {
      const item = document.createElement('div');
      item.className = 'gallery-thumb-item';

      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = file.name;

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'thumb-delete-btn';
      deleteBtn.innerHTML = '&times;';
      deleteBtn.setAttribute('aria-label', `Remove ${file.name}`);
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        state.galleryFiles.splice(idx, 1);
        URL.revokeObjectURL(img.src);
        renderGalleryPreviews();
      });

      item.appendChild(img);
      item.appendChild(deleteBtn);
      el.galleryPreviewGrid.appendChild(item);
    });
  }

  async function handleGalleryEnrollSubmit(e) {
    e.preventDefault();

    const name = el.galleryName.value.trim();
    const role = el.galleryRole.value.trim();
    const notes = el.galleryNotes.value.trim();

    if (!name) {
      showToast('Please enter member name', 'error');
      el.galleryName.focus();
      return;
    }

    if (state.galleryFiles.length === 0) {
      showToast('Please select at least 1 image file', 'error');
      return;
    }

    setButtonLoading(el.btnUploadEnrollGallery, true, 'Uploading & Processing...');

    const formData = new FormData();
    formData.append('name', name);
    formData.append('role', role);
    formData.append('notes', notes);

    // Append files to both 'images' and 'files' to ensure compatibility with backend handler
    state.galleryFiles.forEach((file) => {
      formData.append('images', file);
      formData.append('files', file);
    });

    try {
      const response = await fetch('/api/enroll/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(result.detail || result.error || result.message || `HTTP ${response.status}`);
      }

      showToast(`Successfully enrolled '${name}'!`, 'success', 3500);

      // Clean up
      el.galleryEnrollForm.reset();
      state.galleryFiles = [];
      renderGalleryPreviews();

      // Switch to directory
      switchTab('tab-directory');
      loadDirectory();
    } catch (err) {
      console.error('Gallery upload error:', err);
      showToast(`Upload failed: ${err.message}`, 'error', 4500);
    } finally {
      setButtonLoading(el.btnUploadEnrollGallery, false, 'Upload & Enroll');
    }
  }

  function setButtonLoading(button, isLoading, text) {
    const label = button.querySelector('.btn-label');
    const spinner = button.querySelector('.btn-spinner');

    if (isLoading) {
      button.disabled = true;
      if (spinner) spinner.classList.remove('hidden');
      if (label) label.textContent = text;
    } else {
      button.disabled = false;
      if (spinner) spinner.classList.add('hidden');
      if (label) label.textContent = text;
    }
  }

  // =========================================================================
  // DIRECTORY: FETCH, SEARCH & DELETE
  // =========================================================================
  async function loadDirectory() {
    el.directoryFilterHint.textContent = 'Refreshing...';

    try {
      const response = await fetch('/api/members');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      // Support array directly or object wrapper { members: [...] } / { identities: [...] }
      let members = [];
      if (Array.isArray(data)) {
        members = data;
      } else if (data.members && Array.isArray(data.members)) {
        members = data.members;
      } else if (data.identities && Array.isArray(data.identities)) {
        members = data.identities;
      }

      state.directoryMembers = members;
      renderDirectoryList();
      el.directoryFilterHint.textContent = 'Showing all records';
    } catch (err) {
      console.error('Failed to load directory:', err);
      showToast('Could not fetch members list', 'error', 3000);
      el.directoryFilterHint.textContent = 'Error loading records';
    }
  }

  function renderDirectoryList(filterQuery = '') {
    const query = filterQuery.toLowerCase().trim();

    const filtered = state.directoryMembers.filter((item) => {
      if (!query) return true;
      const name = (item.name || '').toLowerCase();
      const role = (item.role || (item.details && item.details.role) || '').toLowerCase();
      const notes = (item.notes || (item.details && item.details.notes) || '').toLowerCase();
      return name.includes(query) || role.includes(query) || notes.includes(query);
    });

    el.directoryCountBadge.textContent = `${state.directoryMembers.length} Enrolled`;

    if (query) {
      el.directoryFilterHint.textContent = `${filtered.length} of ${state.directoryMembers.length} found`;
    } else {
      el.directoryFilterHint.textContent = 'Showing all records';
    }

    el.directoryList.innerHTML = '';

    if (filtered.length === 0) {
      el.directoryEmptyState.classList.remove('hidden');
      if (state.directoryMembers.length === 0) {
        el.emptyTitle.textContent = 'No Members Enrolled';
        el.emptyDesc.textContent = 'Go to the Enroll tab to capture or upload face identities.';
      } else {
        el.emptyTitle.textContent = 'No Search Results';
        el.emptyDesc.textContent = `No identities match '${query}'.`;
      }
      return;
    }

    el.directoryEmptyState.classList.add('hidden');

    filtered.forEach((member) => {
      const name = member.name || 'Unnamed';
      let role = member.role || (member.details && member.details.role) || 'Identity';
      let notes = member.notes || (member.details && member.details.notes) || '';

      const card = document.createElement('div');
      card.className = 'member-card glass-panel';

      const initials = name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);

      card.innerHTML = `
        <div class="member-info-col">
          <div class="member-avatar">${initials || 'ID'}</div>
          <div class="member-details">
            <h4 class="member-name">${escapeHtml(name)}</h4>
            <div class="member-tags">
              <span class="badge-role">${escapeHtml(role)}</span>
            </div>
            ${notes ? `<p class="member-notes" title="${escapeHtml(notes)}">${escapeHtml(notes)}</p>` : ''}
          </div>
        </div>
        <button type="button" class="member-delete-btn" title="Delete ${escapeHtml(name)}" aria-label="Delete ${escapeHtml(name)}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            <line x1="10" y1="11" x2="10" y2="17"/>
            <line x1="14" y1="11" x2="14" y2="17"/>
          </svg>
        </button>
      `;

      card.querySelector('.member-delete-btn').addEventListener('click', () => {
        promptDeleteMember(name);
      });

      el.directoryList.appendChild(card);
    });
  }

  function promptDeleteMember(name) {
    state.memberToDelete = name;
    el.deleteTargetName.textContent = name;
    el.confirmModal.classList.remove('hidden');
  }

  function closeDeleteModal() {
    state.memberToDelete = null;
    el.confirmModal.classList.add('hidden');
  }

  async function executeDeleteMember() {
    const name = state.memberToDelete;
    if (!name) return;

    closeDeleteModal();
    showToast(`Deleting '${name}'...`, 'info', 1500);

    try {
      const response = await fetch(`/api/members/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(result.detail || result.error || result.message || `HTTP ${response.status}`);
      }

      showToast(`'${name}' deleted successfully`, 'success', 2500);

      // Remove locally and re-render
      state.directoryMembers = state.directoryMembers.filter((m) => m.name !== name);
      renderDirectoryList(el.directorySearch.value);
    } catch (err) {
      console.error('Delete error:', err);
      showToast(`Failed to delete: ${err.message}`, 'error', 4000);
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // =========================================================================
  // TAB NAVIGATION & SCREEN SWITCHING
  // =========================================================================
  function switchTab(targetId) {
    state.currentTab = targetId;

    // Update bottom nav buttons
    el.navTabs.forEach((tab) => {
      const isTarget = tab.dataset.target === targetId;
      tab.classList.toggle('active', isTarget);
      tab.setAttribute('aria-selected', isTarget ? 'true' : 'false');
    });

    // Update screen display
    el.tabScreens.forEach((screen) => {
      const isTarget = screen.id === targetId;
      screen.classList.toggle('active', isTarget);
    });

    // Tab-specific lifecycle logic
    if (targetId === 'tab-live') {
      updateCanvasDimensions();
    } else if (targetId === 'tab-enroll') {
      if (el.enrollVideo && el.enrollVideo.paused && state.cameraStream) {
        el.enrollVideo.play().catch(() => {});
      }
    } else if (targetId === 'tab-directory') {
      loadDirectory();
    }
  }

  function switchEnrollMode(mode) {
    state.enrollMode = mode;
    const isShutter = mode === 'shutter';

    el.segmentShutter.classList.toggle('active', isShutter);
    el.segmentShutter.setAttribute('aria-selected', isShutter ? 'true' : 'false');

    el.segmentGallery.classList.toggle('active', !isShutter);
    el.segmentGallery.setAttribute('aria-selected', !isShutter ? 'true' : 'false');

    el.enrollShutterView.classList.toggle('active', isShutter);
    el.enrollGalleryView.classList.toggle('active', !isShutter);
  }

  // =========================================================================
  // STREAM TOGGLE (PLAY / PAUSE)
  // =========================================================================
  function toggleStreamPlayback() {
    state.isStreamPaused = !state.isStreamPaused;

    if (state.isStreamPaused) {
      el.iconStreamPause.classList.add('hidden');
      el.iconStreamPlay.classList.remove('hidden');
      el.streamStatusText.textContent = 'Paused';
      el.streamFps.textContent = '0.0 FPS';
      overlayCtx.clearRect(0, 0, el.overlayCanvas.width, el.overlayCanvas.height);
      hideRecognizedCard();
      showToast('Recognition stream paused', 'info', 1500);
    } else {
      el.iconStreamPause.classList.remove('hidden');
      el.iconStreamPlay.classList.add('hidden');
      el.streamStatusText.textContent = state.wsConnected ? 'Streaming' : 'Connecting';
      state.isAwaitingResponse = false;
      showToast('Recognition stream resumed', 'info', 1500);
    }
  }

  // =========================================================================
  // EVENT LISTENERS BINDING
  // =========================================================================
  function bindEventListeners() {
    // Bottom Tab Bar
    el.navTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.target;
        if (target) switchTab(target);
      });
    });

    // Live Tab Controls
    el.btnFlipCamera.addEventListener('click', toggleCameraFacing);
    el.btnToggleStream.addEventListener('click', toggleStreamPlayback);
    el.btnCloseCard.addEventListener('click', hideRecognizedCard);

    // Enrollment Studio Segments
    el.segmentShutter.addEventListener('click', () => switchEnrollMode('shutter'));
    el.segmentGallery.addEventListener('click', () => switchEnrollMode('gallery'));

    // Sub-view A: Camera Shutter
    el.btnEnrollFlip.addEventListener('click', toggleCameraFacing);
    el.btnShutter.addEventListener('click', handleSnapShutter);
    el.btnClearShutterPhotos.addEventListener('click', clearAllShutterPhotos);
    el.shutterEnrollForm.addEventListener('submit', handleShutterEnrollSubmit);

    // Sub-view B: Gallery Upload
    el.btnSelectFiles.addEventListener('click', () => el.galleryFileInput.click());
    el.galleryDropzone.addEventListener('click', (e) => {
      if (e.target !== el.btnSelectFiles) {
        el.galleryFileInput.click();
      }
    });

    el.galleryFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleGalleryFilesSelected(e.target.files);
      }
    });

    // Drag and drop support for Gallery Dropzone
    ['dragenter', 'dragover'].forEach((eventName) => {
      el.galleryDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.galleryDropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      el.galleryDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.galleryDropzone.classList.remove('dragover');
      });
    });

    el.galleryDropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleGalleryFilesSelected(e.dataTransfer.files);
      }
    });

    el.galleryEnrollForm.addEventListener('submit', handleGalleryEnrollSubmit);

    // Directory
    el.directorySearch.addEventListener('input', (e) => {
      const q = e.target.value;
      el.btnClearSearch.classList.toggle('hidden', !q);
      renderDirectoryList(q);
    });

    el.btnClearSearch.addEventListener('click', () => {
      el.directorySearch.value = '';
      el.btnClearSearch.classList.add('hidden');
      renderDirectoryList('');
    });

    el.btnRefreshDirectory.addEventListener('click', () => {
      loadDirectory();
      showToast('Directory refreshed', 'info', 1200);
    });

    // Delete Confirmation Modal
    el.btnCancelDelete.addEventListener('click', closeDeleteModal);
    el.btnConfirmDelete.addEventListener('click', executeDeleteMember);
    el.confirmModal.addEventListener('click', (e) => {
      if (e.target === el.confirmModal) closeDeleteModal();
    });

    // Visibility / backgrounding optimization
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        state.isAwaitingResponse = false;
      }
    });
  }

  // =========================================================================
  // BOOTSTRAP INITIALIZATION
  // =========================================================================
  async function initializeApp() {
    console.log('[App] Initializing FACE 2.0 Mobile Client...');
    bindEventListeners();

    // 1. Start Camera (defaults to rear/environment camera)
    await initCamera('environment');

    // 2. Connect WebSocket to /ws/recognize
    connectWebSocket();

    // 3. Kick off live stream loop with backpressure control
    scheduleStreamLoop();

    // 4. Preload directory records
    loadDirectory();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
  } else {
    initializeApp();
  }
})();
