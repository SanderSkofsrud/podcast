'use client';
import React, { useState, useEffect } from 'react';

const AudioUploader: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [status, setStatus] = useState<string | null>(null);
    const [requestId, setRequestId] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setFile(event.target.files ? event.target.files[0] : null);
        setDownloadUrl(null);
        setStatus(null);
        setRequestId(null);
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);

        const formData = new FormData();
        formData.append('audio', file);

        try {
            const response = await fetch('http://localhost:5001/process_audio', {
                method: 'POST',
                body: formData,
            });

            console.log(response);  // Log response for debugging

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                setDownloadUrl(url);
            } else {
                alert('Failed to process audio');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('An error occurred while uploading the file.');
        } finally {
            setLoading(false);
        }
    };


    const pollStatus = (id: string) => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`http://localhost:5001/status/${id}`);
                const data = await res.json();
                setStatus(data.status);

                if (data.status === "Completed" && requestId) {
                    const response = await fetch(`http://localhost:5001/download/${requestId}`);
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        setDownloadUrl(url);
                        clearInterval(interval); // Stop polling when completed
                    }
                } else if (data.status === "Error") {
                    alert("An error occurred during processing.");
                    clearInterval(interval);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                clearInterval(interval);
            }
        }, 2000); // Poll every 2 seconds
    };

    return (
        <div style={{ padding: '20px', maxWidth: '400px', margin: 'auto' }}>
            <h2>Upload Audio for Ad Removal</h2>
            <input type="file" accept="audio/*" onChange={handleFileChange} />
            <button onClick={handleUpload} disabled={!file || loading} style={{ marginTop: '10px' }}>
                {loading ? 'Processing...' : 'Upload and Process'}
            </button>

            {status && <p>Status: {status}</p>}

            {downloadUrl && (
                <div style={{ marginTop: '20px' }}>
                    <a href={downloadUrl} download="edited_audio.mp3">
                        Download Processed Audio
                    </a>
                </div>
            )}
        </div>
    );
};

export default AudioUploader;
