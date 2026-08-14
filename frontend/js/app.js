const API_BASE = window.location.origin;

const HarvestLenz = {
  get token() { return localStorage.getItem('hl_token'); },
  set token(v) { v ? localStorage.setItem('hl_token', v) : localStorage.removeItem('hl_token'); },
  get refreshToken() { return localStorage.getItem('hl_refresh'); },
  set refreshToken(v) { v ? localStorage.setItem('hl_refresh', v) : localStorage.removeItem('hl_refresh'); },
  get user() { try { return JSON.parse(localStorage.getItem('hl_user')); } catch { return null; } },
  set user(v) { v ? localStorage.setItem('hl_user', JSON.stringify(v)) : localStorage.removeItem('hl_user'); },
  get scans() { try { return JSON.parse(localStorage.getItem('hl_scans') || '[]'); } catch { return []; } },
  set scans(v) { localStorage.setItem('hl_scans', JSON.stringify(v)); },

  isLoggedIn() { return !!this.token; },

  requireAuth() {
    if (!this.isLoggedIn()) { window.location.href = '/frontend/login.html'; return false; }
    return true;
  },

  async api(path, opts = {}) {
    const headers = { ...opts.headers };
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
    if (opts.json) {
      headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });
    if (res.status === 401) {
      this.logout();
      window.location.href = '/frontend/login.html';
      throw new Error('Session expired');
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Request failed');
    }
    if (res.headers.get('content-type')?.includes('application/json')) return res.json();
    return res;
  },

  async register(data) {
    const res = await this.api('/auth/register', { method: 'POST', json: data });
    this.user = res;
    return res;
  },

  async login(email, password) {
    const res = await this.api('/auth/login', { method: 'POST', json: { email, password } });
    this.token = res.access_token;
    this.refreshToken = res.refresh_token;
    const me = await this.api('/auth/me');
    this.user = me;
    return me;
  },

  logout() {
    this.token = null;
    this.refreshToken = null;
    this.user = null;
  },

  async analyzeFruit(file, fruitType, isSingle = false) {
    const fd = new FormData();
    fd.append('file', file);
    return this.api(`/analyze?fruit_type=${encodeURIComponent(fruitType)}&is_single=${isSingle}`, {
      method: 'POST',
      body: fd,
    });
  },

  async getReport(scanId) {
    return this.api(`/scan/report/${scanId}`);
  },

  async getPassport(fruitId) {
    return this.api(`/scan/passport/${fruitId}`);
  },

  async getAnalysis(sessionId) {
    return this.api(`/analysis/${sessionId}`);
  },

  async getAnalysisStatus(sessionId) {
    return this.api(`/analysis/${sessionId}/status`);
  },

  saveScan(session) {
    const scans = this.scans;
    const idx = scans.findIndex(s => s.session_id === session.session_id);
    if (idx >= 0) scans[idx] = session; else scans.unshift(session);
    if (scans.length > 50) scans.length = 50;
    this.scans = scans;
  },

  getScan(id) { return this.scans.find(s => s.session_id === id); },

  gradeClass(grade) {
    const g = (grade || '').toLowerCase();
    if (g === 'good' || g === 'premium') return 'badge-good';
    if (g === 'better') return 'badge-better';
    if (g === 'medium') return 'badge-medium';
    if (g === 'reject') return 'badge-reject';
    return '';
  }
};

function toast(msg, type = 'success') {
  let c = document.querySelector('.toast-container');
  if (!c) { c = document.createElement('div'); c.className = 'toast-container'; document.body.appendChild(c); }
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function showLoading(msg = 'Processing...') {
  let o = document.getElementById('global-loading');
  if (!o) {
    o = document.createElement('div');
    o.id = 'global-loading';
    o.className = 'loading-overlay';
    o.innerHTML = `<div class="loading-box"><div class="spinner"></div><p id="loading-msg"></p></div>`;
    document.body.appendChild(o);
  }
  document.getElementById('loading-msg').textContent = msg;
  o.style.display = 'flex';
}

function hideLoading() {
  const o = document.getElementById('global-loading');
  if (o) o.style.display = 'none';
}

function formatDate(d) {
  if (!d) return '';
  return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function qs(key) { return new URLSearchParams(window.location.search).get(key); }
