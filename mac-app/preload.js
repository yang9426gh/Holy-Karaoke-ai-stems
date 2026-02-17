const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('holy', {
  install: () => ipcRenderer.invoke('install'),
  start: () => ipcRenderer.invoke('start'),
  onLog: (cb) => {
    ipcRenderer.removeAllListeners('log');
    ipcRenderer.on('log', (_ev, msg) => cb(msg));
  },
  onPhase: (cb) => {
    ipcRenderer.removeAllListeners('phase');
    ipcRenderer.on('phase', (_ev, payload) => cb(payload));
  },
  getConfig: () => ipcRenderer.invoke('config.get'),
  setConfig: (patch) => ipcRenderer.invoke('config.set', patch),
  saveFile: (url, suggestedName) => ipcRenderer.invoke('file.save', { url, suggestedName })
});
