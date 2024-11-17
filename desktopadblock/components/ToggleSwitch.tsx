// src/components/ToggleSwitch.tsx

import React from 'react';

interface ToggleSwitchProps {
  mode: 'fast' | 'accurate';
  onToggle: (mode: 'fast' | 'accurate') => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ mode, onToggle }) => {
  const isAccurate = mode === 'accurate';

  const handleToggle = () => {
    const newMode = isAccurate ? 'fast' : 'accurate';
    onToggle(newMode);
  };

  return (
    <div className="flex items-center space-x-4">
      <span
        className={`text-gray-700 font-medium ${
          mode === 'fast' ? 'font-bold text-blue-600' : ''
        }`}
      >
        Fast
      </span>
      <label htmlFor="modeToggle" className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          id="modeToggle"
          className="sr-only peer"
          checked={isAccurate}
          onChange={handleToggle}
        />
        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer dark:bg-gray-600 peer-checked:bg-blue-600 transition-colors duration-300"></div>
        <span
          className={`absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-300 transform ${
            isAccurate ? 'translate-x-full bg-blue-500' : ''
          }`}
        ></span>
      </label>
      <span
        className={`text-gray-700 font-medium ${
          mode === 'accurate' ? 'font-bold text-blue-600' : ''
        }`}
      >
        Accurate
      </span>
    </div>
  );
};

export default ToggleSwitch;
