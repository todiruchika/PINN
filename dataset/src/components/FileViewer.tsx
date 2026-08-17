/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { FolderCode, FileCode, Copy, Check, Terminal } from "lucide-react";
import { PIPELINE_FILES } from "../utils/codeTemplates";
import { PipelineCodeFile } from "../types";

export default function FileViewer() {
  const [selectedFile, setSelectedFile] = useState<PipelineCodeFile>(PIPELINE_FILES[0]);
  const [copied, setCopied] = useState<boolean>(false);
  const [currentFileContent, setCurrentFileContent] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Modern browser-safe text fetch for the workspace file
  const handleLoadFullCodeFile = async (file: PipelineCodeFile) => {
    setLoading(true);
    try {
      // In default setup we put all generated deliverables inside /py_pipeline/
      const response = await fetch(`/py_pipeline/${file.filename}`);
      if (response.ok) {
        const text = await response.text();
        setCurrentFileContent(text);
      } else {
        // Fallback to descriptive run command if file loading is restricted
        setCurrentFileContent(`# Run command to run script in terminal:\n${file.content}\n\n# Note: This file is fully written at "/py_pipeline/${file.filename}" in your workspace.\n# You can view it in the file manager sidebar!`);
      }
    } catch {
      setCurrentFileContent(`# Run command: \` ${file.content} \``);
    } finally {
      setSelectedFile(file);
      setLoading(false);
    }
  };

  React.useEffect(() => {
    handleLoadFullCodeFile(PIPELINE_FILES[0]);
  }, []);

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden backdrop-blur-md" id="pipeline-file-viewer">
      {/* Tab bar */}
      <div className="bg-zinc-950 px-4 py-3 border-b border-zinc-800 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <FolderCode className="w-5 h-5 text-amber-500" />
          <span className="font-mono text-xs font-semibold tracking-wider uppercase text-zinc-300">
            Pipeline Codebase Explorer
          </span>
        </div>
        <span className="text-[10px] font-mono text-zinc-500 bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded">
          /py_pipeline/
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 min-h-[450px]">
        {/* Left Column: Explorer Tab */}
        <div className="md:col-span-4 border-r border-zinc-800 bg-zinc-950/40 p-2 space-y-1 max-h-[500px] overflow-y-auto">
          {PIPELINE_FILES.map((file) => {
            const isSelected = selectedFile.filename === file.filename;
            return (
              <button
                key={file.filename}
                onClick={() => handleLoadFullCodeFile(file)}
                className={`w-full text-left p-3 rounded-lg font-mono transition-all duration-150 border ${
                  isSelected
                    ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                    : "bg-transparent border-transparent hover:bg-zinc-800/40 hover:border-zinc-805 text-zinc-400"
                }`}
              >
                <div className="flex items-center space-x-2 text-xs font-semibold">
                  <FileCode className={`w-4 h-4 ${isSelected ? "text-amber-400" : "text-zinc-500"}`} />
                  <span>{file.filename}</span>
                </div>
                <p className="text-[10px] text-zinc-500 mt-1 leading-snug line-clamp-2">
                  {file.description}
                </p>
              </button>
            );
          })}
        </div>

        {/* Right Column: Code viewer layout */}
        <div className="md:col-span-8 flex flex-col bg-zinc-950/20 max-h-[500px]">
          {/* File Header */}
          <div className="flex justify-between items-center bg-zinc-900/40 px-4 py-2 border-b border-zinc-850">
            <div className="flex flex-col">
              <span className="text-xs font-semibold text-zinc-300 font-mono text-amber-500">
                {selectedFile.name}
              </span>
              <span className="text-[10px] font-mono text-zinc-500">
                Location: /py_pipeline/{selectedFile.filename}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleCopyCode(currentFileContent)}
                className="flex items-center space-x-1 px-3 py-1 bg-zinc-800 border border-zinc-700 hover:bg-zinc-750 rounded text-[11px] text-zinc-300 transition"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-lime-500" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Copy Code</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Code output area */}
          <div className="flex-1 p-4 overflow-auto font-mono text-xs text-zinc-300 leading-relaxed max-h-[420px] bg-zinc-950/65 select-text">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 space-y-2">
                <div className="w-6 h-6 border-2 border-t-amber-500 border-zinc-700 rounded-full animate-spin"></div>
                <span className="text-[10px] text-zinc-500 tracking-wider font-mono">LOADING FILE CORE...</span>
              </div>
            ) : (
              <pre className="whitespace-pre overflow-x-auto selection:bg-amber-500/20">
                <code>{currentFileContent}</code>
              </pre>
            )}
          </div>

          {/* Quick instructions snippet CLI footer */}
          <div className="bg-zinc-900/60 p-3 border-t border-zinc-850 flex items-center space-x-2 text-[10px] font-mono text-zinc-400">
            <Terminal className="w-3.5 h-3.5 text-amber-500/70" />
            <span className="font-semibold text-zinc-500 uppercase">CLI Command:</span>
            <code className="bg-zinc-950 px-2 py-0.5 border border-zinc-800 rounded text-zinc-300">
              {selectedFile.content}
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
