"use client";
import React, { useState } from 'react';
import AudioUploader from "@/components/audioUploader";

const Home: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

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
      setFile(files[0]);
    }
  };

  const removeFile = () => {
    setFile(null);
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

      {isDragging && (
        <div className="fixed inset-0 bg-purple-800 bg-opacity-50 backdrop-blur-md flex items-center justify-center z-50">
          <p className="text-white text-2xl font-medium animate-pulse">
            Drop file here
          </p>
        </div>
      )}

      <div
        className={`border-4 border-dashed rounded-xl p-16 mx-auto max-w-2xl transition-colors duration-300 ${
          isDragging ? 'border-purple-600 bg-purple-100' : 'border-gray-400 bg-white'
        } shadow-lg`}
      >
        <p className="text-gray-700 font-medium">
          Drag file here or click to upload
        </p>
      </div>

      <div className="mt-10 w-full max-w-lg">
        <AudioUploader onFileSelected={handleFileSelected} file={file} />
      </div>

      {file && (
        <div className="mt-8 w-full max-w-lg bg-white rounded-lg shadow-md p-5">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">file:</h2>
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
