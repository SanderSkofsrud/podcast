// src/components/AudioUploader.tsx

'use client';
import React, { useState } from 'react';

interface AudioUploaderProps {
  onFileSelected: (file: File | null) => void;
  file: File | null;
  mode: 'fast' | 'accurate';
}

const AudioUploader: React.FC<AudioUploaderProps> = ({ onFileSelected, file, mode }) => {
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files ? event.target.files[0] : null;
    onFileSelected(selectedFile);
    setDownloadUrl(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('mode', mode);

    try {
      const response = await fetch('http://localhost:5001/process_audio', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        setDownloadUrl(url);
      } else {
        const errorData = await response.json();
        alert(`Failed to process audio: ${errorData.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred while uploading the file.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-xl mx-auto bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-gray-800 mb-5">Upload File to Remove Ads</h2>

      <input
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="block w-full px-4 py-3 text-gray-700 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 transition"
      />
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className={`mt-5 w-full py-3 text-white font-semibold rounded-lg transition ${
          !file || loading
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-purple-600 hover:bg-purple-700 shadow-md'
        }`}
      >
        {loading ? (
          <span className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
        ) : (
          'Upload'
        )}
      </button>

      <p className="mt-2 text-gray-500">
        Processing Mode: <span className="font-medium capitalize">{mode}</span>
      </p>

      {downloadUrl && (
        <div className="mt-6">
          <a
            href={downloadUrl}
            download="edited_audio.mp3"
            className="inline-block px-4 py-2 bg-purple-100 text-purple-700 rounded-lg font-medium hover:bg-purple-200 transition w-full text-center"
          >
            Download Ad-Free Audio
          </a>
        </div>
      )}
    </div>
  );
};

export default AudioUploader;
