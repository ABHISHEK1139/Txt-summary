// DOM References
const chatInput = document.getElementById('chat-input');
const fileInput = document.getElementById('file-upload');
const history = document.getElementById('chat-history');

let controller = null; // AbortController

// --- Utilities ---

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

function toggleState(isGenerating) {
    document.getElementById('send-btn').style.display = isGenerating ? 'none' : 'flex';
    document.getElementById('stop-btn').style.display = isGenerating ? 'flex' : 'none';
    if (isGenerating) {
        document.getElementById('welcome-screen').style.display = 'none';
    }
}

function stopGeneration() {
    if (controller) {
        controller.abort();
        controller = null;
        toggleState(false);
        appendMessage('ai', '<em>Generation stopped by user.</em>');
    }
}

function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    let html = '';
    if (role === 'ai') {
        html = `
            <div class="avatar"><span class="material-icons-round">smart_toy</span></div>
            <div class="bubble markdown-body">
                ${content}
                <button class="copy-btn" onclick="copyText(this)" title="Copy">
                    <span class="material-icons-round">content_copy</span>
                </button>
            </div>
        `;
    } else {
        html = `<div class="bubble">${content}</div>`;
    }

    div.innerHTML = html;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
    return div.querySelector('.bubble');
}

function copyText(btn) {
    const bubble = btn.parentElement;
    const text = bubble.innerText.replace('content_copy', '').replace('check', '').trim();
    navigator.clipboard.writeText(text).then(() => {
        const icon = btn.querySelector('span');
        icon.innerText = 'check';
        setTimeout(() => icon.innerText = 'content_copy', 2000);
    });
}

// --- Event Listeners ---

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
});

// --- Chat Handler ---

async function handleSend() {
    const text = chatInput.value.trim();
    if (!text) return;

    toggleState(true);
    appendMessage('user', text);
    chatInput.value = '';
    autoResize(chatInput);

    const aiBubble = appendMessage('ai', '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>');

    controller = new AbortController();
    const signal = controller.signal;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
            signal: signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        aiBubble.innerHTML = ''; // Clear loader

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value);
            aiBubble.innerHTML = marked.parse(fullText);
            history.scrollTop = history.scrollHeight;
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            // Handled in stopGeneration
        } else {
            aiBubble.innerText = "Error: " + e.message;
        }
    } finally {
        toggleState(false);
        controller = null;
    }
}

// --- File Upload Handler ---

async function handleFileUpload(file) {
    fileInput.value = '';

    // File size check (25MB max)
    if (file.size > 25 * 1024 * 1024) {
        appendMessage('user', `<div class="file-tag"><span class="material-icons-round">description</span> ${file.name}</div>`);
        appendMessage('ai', '⚠️ File too large (max 25MB). Try a smaller document.');
        return;
    }

    toggleState(true);

    const userMsg = `<div class="file-tag"><span class="material-icons-round">description</span> ${file.name}</div> Summarize this document.`;
    appendMessage('user', userMsg);

    const aiBubble = appendMessage('ai', '📄 Reading document... <div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>');

    let form = new FormData();
    form.append("file", file);

    controller = new AbortController();
    const signal = controller.signal;

    try {
        let res = await fetch("/upload", { method: "POST", body: form, signal: signal });

        // Check for HTTP errors first
        if (!res.ok && !res.headers.get('content-type')?.includes('ndjson')) {
            const errData = await res.json().catch(() => ({ error: 'Server error' }));
            aiBubble.innerText = '❌ ' + (errData.error || `Server error (${res.status})`);
            return;
        }

        const contentType = res.headers.get('content-type') || '';

        if (contentType.includes('application/x-ndjson')) {
            // Streaming NDJSON — long document with progressive section rendering
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let totalSections = 0;
            let renderedSections = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const msg = JSON.parse(line);
                        if (msg.type === 'info') {
                            totalSections = msg.total_sections;
                            aiBubble.innerHTML = `<div style="color:var(--accent);margin-bottom:8px">📄 Analyzing <strong>${msg.word_count.toLocaleString()}</strong> words across <strong>${totalSections}</strong> sections...</div><div class="progress-bar"><div class="progress-fill" style="width:0%"></div></div>`;
                        } else if (msg.type === 'progress') {
                            const pct = Math.round((msg.section / totalSections) * 100);
                            const summary = msg.section_summary;
                            if (summary && !summary.startsWith('(')) {
                                const heading = msg.section_heading;
                                const num = msg.section_number;
                                const label = heading ? `**${num}. ${heading}:** ${summary}` : `**${num}.** ${summary}`;
                                renderedSections.push(label);
                            }

                            // Build live markdown
                            let liveMarkdown = `📄 **Analyzing...** (${msg.section}/${totalSections} sections)\n\n---\n\n`;
                            liveMarkdown += renderedSections.join('\n\n');

                            aiBubble.innerHTML = marked.parse(liveMarkdown);

                            // Animate the last added section
                            const allPs = aiBubble.querySelectorAll('p, strong');
                            if (allPs.length > 0) {
                                const last = allPs[allPs.length - 1];
                                last.style.animation = 'fadeSlideIn 0.4s ease-out';
                            }
                        } else if (msg.type === 'complete') {
                            // Final: show overall summary + all sections
                            aiBubble.innerHTML = marked.parse(msg.summary);
                            aiBubble.style.animation = 'fadeSlideIn 0.5s ease-out';
                        } else if (msg.type === 'error') {
                            aiBubble.innerText = '❌ ' + msg.message;
                        }
                    } catch (e) { /* skip bad JSON */ }
                }
                history.scrollTop = history.scrollHeight;
            }

            // If no complete message was received, show what we have
            if (renderedSections.length > 0 && !aiBubble.innerHTML.includes('Key Points')) {
                let fallback = '📄 **Partial Summary** (connection interrupted)\n\n---\n\n';
                fallback += renderedSections.join('\n\n');
                aiBubble.innerHTML = marked.parse(fallback);
            }
        } else {
            // Short document — direct JSON response
            let data = await res.json();
            if (res.ok) {
                aiBubble.innerHTML = marked.parse(data.summary);
            } else {
                aiBubble.innerText = 'Error: ' + (data.error || 'Unknown');
            }
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            // Handled in stopGeneration
        } else {
            aiBubble.innerText = 'Network Error: ' + e.message;
        }
    } finally {
        toggleState(false);
        controller = null;
    }
    history.scrollTop = history.scrollHeight;
}

// --- About Modal Functions ---

function openAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) modal.classList.add('active');
}

function closeAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) modal.classList.remove('active');
}

function handleModalOverlayClick(event) {
    if (event.target.id === 'about-modal') {
        closeAboutModal();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAboutModal();
    }
});

