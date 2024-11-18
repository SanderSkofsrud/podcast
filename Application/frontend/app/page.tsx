// src/pages/Home.tsx

"use client";
import React, { useState } from 'react';
import AudioUploader from "@/components/audioUploader";
import ToggleSwitch from "@/components/ToggleSwitch";

const Home: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<'fast' | 'accurate'>('fast');

  const handleFileSelected = (selectedFile: File | null) => {
    setFile(selectedFile);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (e.relatedTarget === null) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const droppedFile = files[0];
      setFile(droppedFile);
    }
  };

  const removeFile = () => {
    setFile(null);
  };

  const handleToggle = (selectedMode: 'fast' | 'accurate') => {
    setMode(selectedMode);
  };

  return (
    <div
      className="relative min-h-screen bg-gray-100 p-10 flex flex-col items-center"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h1 className="text-4xl font-extrabold text-gray-800 mb-8">
        Ad Remover for Podcasts
      </h1>
      <ToggleSwitch mode={mode} onToggle={handleToggle} />

      {isDragging && (
        <div className="fixed inset-0 bg-purple-800 bg-opacity-50 backdrop-blur-md flex items-center justify-center z-40">
          <div className="absolute inset-0 border-4 border-dotted border-white rounded-lg m-4"></div>
          <div className="absolute inset-0 bg-white opacity-50"></div>
          <p className="text-white text-8xl font-medium animate-pulse z-50">
            Drop file here
          </p>
        </div>
      )}

      <div className="mt-10 w-full max-w-xl">
        {/* Pass the updated file state and mode to the AudioUploader */}
        <AudioUploader onFileSelected={handleFileSelected} file={file} mode={mode} />
      </div>

      {file && (
        <div className="mt-8 w-full max-w-xl bg-white rounded-lg shadow-md p-5">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">File:</h2>
          <div className="flex items-center justify-between">
            <span className="truncate">{file.name}</span>
            <button
              onClick={removeFile}
              className="ml-4 px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition duration-200"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
