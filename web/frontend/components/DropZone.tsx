'use client';

import { useCallback, useRef, useState } from 'react';

export default function DropZone({
  accept,
  onFile,
  fileName,
  hint,
}: {
  accept: string;
  onFile: (file: File) => void;
  fileName?: string | null;
  hint: string;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (files && files[0]) onFile(files[0]);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-all duration-200 ${
        dragOver
          ? 'scale-[1.02] border-accent bg-accent/5 shadow-glow'
          : 'border-borderLt bg-card hover:scale-[1.01] hover:border-accent/60'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div
        className={`mb-2 text-3xl transition-transform duration-300 ${dragOver ? 'scale-125 float-y' : 'group-hover:scale-110'}`}
      >
        📤
      </div>
      {fileName ? (
        <div className="text-sm font-medium text-accent">{fileName}</div>
      ) : (
        <div className="text-sm font-medium text-textMain">
          Click or drag a file here
        </div>
      )}
      <div className="mt-1 text-xs text-textDim">{hint}</div>
    </div>
  );
}
