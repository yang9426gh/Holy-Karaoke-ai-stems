'use client';

import { useEffect, useMemo, useState } from 'react';
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type UiSettings } from './settings';

export function useSettings() {
  const [settings, setSettings] = useState<UiSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  const api = useMemo(() => {
    return {
      settings,
      setSettings: (next: UiSettings) => {
        setSettings(next);
        saveSettings(next);
      },
      patch: (patch: Partial<UiSettings>) => {
        setSettings(prev => {
          const next = { ...prev, ...patch };
          saveSettings(next);
          return next;
        });
      },
      reset: () => {
        setSettings(DEFAULT_SETTINGS);
        saveSettings(DEFAULT_SETTINGS);
      }
    };
  }, [settings]);

  return api;
}
