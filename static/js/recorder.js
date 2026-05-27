import { els } from './state.js';
import { askQuestion } from './ask.js';
import { setVoiceStatus } from './speech.js';

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordButton = null;

function getSupportedMimeType() {
  const types = [
    'audio/webm',
    'audio/webm;codecs=opus',
    'audio/ogg',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return 'audio/webm';
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setVoiceStatus('当前环境不支持录音');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = getSupportedMimeType();
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      handleRecordingStop(mimeType);
    };

    mediaRecorder.onerror = () => {
      stream.getTracks().forEach((track) => track.stop());
      setRecordButtonState('idle');
      setVoiceStatus('录音出错');
    };

    mediaRecorder.start();
    isRecording = true;
    setRecordButtonState('recording');
    setVoiceStatus('正在录音…');
  } catch (err) {
    setVoiceStatus('无法访问麦克风');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  setRecordButtonState('idle');
  setVoiceStatus('');
}

async function handleRecordingStop(mimeType) {
  if (!audioChunks.length) {
    setVoiceStatus('录音为空');
    return;
  }
  const blob = new Blob(audioChunks, { type: mimeType });
  audioChunks = [];

  setRecordButtonState('processing');
  setVoiceStatus('正在识别…');

  try {
    const text = await uploadAudio(blob);
    if (text) {
      els.questionInput.value = text;
      askQuestion();
    } else {
      setVoiceStatus('未能识别语音');
    }
  } catch (err) {
    setVoiceStatus('识别失败');
  } finally {
    setRecordButtonState('idle');
  }
}

async function uploadAudio(blob) {
  const response = await fetch('/api/asr', {
    method: 'POST',
    headers: {
      'Content-Type': blob.type || 'audio/webm',
    },
    body: blob,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.text || '';
}

function setRecordButtonState(state) {
  if (!recordButton) return;
  recordButton.dataset.state = state;
  const label = recordButton.querySelector('.record-label');
  if (label) {
    label.textContent = state === 'recording' ? '结束' : '说话';
  }
}

export function initRecorder(button) {
  recordButton = button;
  if (!recordButton) return;
  recordButton.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}
