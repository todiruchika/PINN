/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { 
  Play, StopCircle, Download, CheckCircle, AlertTriangle, Terminal, Cpu, FileSpreadsheet, Server, FileBox 
} from "lucide-react";
import { generateRandomFloorplan, solveHeatEquation, extractSensorsGroup } from "../utils/solver";

interface LogLine {
  text: string;
  type: "info" | "success" | "warning" | "error" | "pde";
}

export default function Console() {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [trainSize, setTrainSize] = useState<number>(500);
  const [valSize, setValSize] = useState<number>(100);
  const [testSize, setTestSize] = useState<number>(100);
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const [finished, setFinished] = useState<boolean>(false);
  
  // Accumulated metadata for download
  const [generatedRows, setGeneratedRows] = useState<any[]>([]);

  const terminalEndRef = useRef<HTMLDivElement>(null);

  const addLogLine = (text: string, type: "info" | "success" | "warning" | "error" | "pde" = "info") => {
    setLogLines((prev) => [...prev, { text: `[${new Date().toLocaleTimeString()}] ${text}`, type }]);
  };

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logLines]);

  const handleStopCompilation = () => {
    setIsRunning(false);
    addLogLine("Dataset generation aborted by user.", "error");
  };

  const handleStartCompilation = () => {
    setIsRunning(true);
    setFinished(false);
    setProgress(0);
    setGeneratedRows([]);
    setLogLines([]);

    addLogLine("INITIALIZING MULTIPROCESSING ORCHESTRATION ENGINE...", "info");
    addLogLine("Checking platform capabilities...", "info");
    addLogLine("Verified FEniCS/DOLFINx libraries: OK", "success");
    addLogLine(`Targeting parallel scale: ${trainSize} train, ${valSize} val, ${testSize} test`, "info");
    addLogLine("Preparing directories structure: dataset/train/, dataset/val/, dataset/test/ ...", "info");

    let currentStep = 0;
    const totalSteps = 40; // Simulated timeline steps
    const metadataRecords: any[] = [];

    const interval = setInterval(() => {
      currentStep++;
      const currentProgress = Math.round((currentStep / totalSteps) * 100);
      setProgress(currentProgress);

      const targetSplit = currentStep <= 22 ? "train" : currentStep <= 31 ? "val" : "test";
      const fileId = 1000 + currentStep;

      // Log sequential phases
      if (currentStep === 1) {
        addLogLine("Step 1: Outlining dataset split boundaries...", "info");
      } else if (currentStep === 3) {
        addLogLine("[POOL-01] Spawning 12 Python multiprocessing worker threads...", "info");
      } else if (currentStep === 6) {
        addLogLine(`[POOL-01] Running floorplan coordinates calculations [Split: ${targetSplit.toUpperCase()}]`, "pde");
      } else if (currentStep === 10) {
        addLogLine("[POOL-01] Solving steady-state heat boundary equations via Dirichlet Laplacian...", "pde");
      } else if (currentStep === 15) {
        addLogLine("Checking LAPACK solver residuals: CONVERGED (residual value: 3.4e-6)", "success");
      } else if (currentStep === 20) {
        addLogLine("Exporting Dense NumPy matrix objects (.npy) scale 128x128...", "info");
      } else if (currentStep === 23) {
        addLogLine("[POOL-02] Starting Validation audits on Train split folder...", "info");
        addLogLine("AUDIT PASSED: No NaN/Inf elements detected.", "success");
        addLogLine("AUDIT PASSED: Thermal diffuse gradient flows outward.", "success");
      } else if (currentStep === 28) {
        addLogLine("Deploying randomized sparse sensors (5, 8, 10 counts) ...", "pde");
      } else if (currentStep === 32) {
        addLogLine("[POOL-03] Validation audits triggered for Val split folder...", "info");
        addLogLine("AUDIT PASSED: Sensor arrays physical Celsius matching strict < 1e-4", "success");
      } else if (currentStep === 36) {
        addLogLine("Processing Test split archives... Saving images/ and sensors/ ...", "info");
      }

      // Generate randomized mini-data record for CSV spreadsheet
      if (currentStep % 2 === 0 && currentStep < 38) {
        const k = 100 + Math.random() * 50;
        const amb = 20 + Math.random() * 20;
        const count = 1 + Math.floor(Math.random() * 9);
        const pk = 65 + Math.random() * 50;
        
        metadataRecords.push({
          sample_id: `thermal_${String(fileId).padStart(5, "0")}`,
          hotspot_count: count,
          thermal_conductivity: k.toFixed(2),
          boundary_temperature: amb.toFixed(2),
          peak_temperature: pk.toFixed(2),
          split: targetSplit
        });
      }

      if (currentStep >= totalSteps) {
        clearInterval(interval);
        setIsRunning(false);
        setFinished(true);
        setGeneratedRows(metadataRecords);

        addLogLine("Compiling consolidated dataset tables metadata.csv...", "info");
        addLogLine("RUNNING FINAL QUALITY ASSESSMENT AND THERMODYNAMIC AUDITS...", "info");
        addLogLine("Quality Assurance Report: ALL METRICS PASSED PHYSICAL BOUNDS.", "success");
        addLogLine("==================================================", "success");
        addLogLine("PINN CHEMICAL/THERMAL DATASET BUILT SUCCESSFULLY!", "success");
        addLogLine(`Total samples generated: ${trainSize + valSize + testSize}. Saved in 'dataset/'`, "success");
        addLogLine("==================================================", "success");
      }
    }, 180);
  };

  // Helper function to synthesize and download compiled metadata.csv
  const handleDownloadCsv = () => {
    if (generatedRows.length === 0) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "sample_id,hotspot_count,thermal_conductivity,boundary_temperature,peak_temperature,split\r\n";

    generatedRows.forEach((row) => {
      csvContent += `${row.sample_id},${row.hotspot_count},${row.thermal_conductivity},${row.boundary_temperature},${row.peak_temperature},${row.split}\r\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "thermal_metadata_simulate.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-5" id="pipeline-orchestrator-console">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-zinc-800 pb-4 mb-5 gap-3">
        <div>
          <h3 className="text-sm font-semibold text-zinc-200 flex items-center space-x-2 font-display">
            <Server className="w-4 h-4 text-emerald-400" />
            <span>Dataset Pipeline Prototyping Console</span>
          </h3>
          <p className="text-[11px] text-zinc-500 mt-0.5 leading-snug">
            Trigger a simulated local multi-threaded generation compile script inside the preview sandbox environment.
          </p>
        </div>

        {/* size input preset */}
        <div className="flex items-center space-x-2 bg-zinc-950 p-1 border border-zinc-850 rounded-lg text-xs font-mono">
          <div className="px-2 py-1 text-zinc-500 font-semibold uppercase text-[10px]">
            Size Profile:
          </div>
          <button
            onClick={() => {
              setTrainSize(20);
              setValSize(5);
              setTestSize(5);
            }}
            disabled={isRunning}
            className={`px-2.5 py-1 rounded text-[10px] uppercase font-bold transition ${
              trainSize === 20
                ? "bg-zinc-805 text-amber-500 border border-zinc-700"
                : "text-zinc-500 hover:text-zinc-400 disabled:opacity-50"
            }`}
          >
            Light Test (30)
          </button>
          <button
            onClick={() => {
              setTrainSize(500);
              setValSize(100);
              setTestSize(100);
            }}
            disabled={isRunning}
            className={`px-2.5 py-1 rounded text-[10px] uppercase font-bold transition ${
              trainSize === 500
                ? "bg-zinc-805 text-amber-500 border border-zinc-700"
                : "text-zinc-500 hover:text-zinc-400 disabled:opacity-50"
            }`}
          >
            Prototype (700)
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-5">
        {/* Configuration sliders column */}
        <div className="md:col-span-4 space-y-4">
          <div className="bg-zinc-950/50 border border-zinc-850 rounded-lg p-3.5 space-y-3 font-mono text-xs">
            <div className="flex items-center space-x-1.5 text-[10px] uppercase text-zinc-500 font-bold border-b border-zinc-900 pb-1.5 mb-2">
              <Cpu className="w-3.5 h-3.5 text-zinc-400" />
              <span>Job Properties Configuration</span>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-zinc-400">
                <span>Train Sub-split size:</span>
                <span className="text-zinc-300 font-semibold">{trainSize} files</span>
              </div>
              <input
                type="range"
                min="10"
                max="1000"
                step="10"
                disabled={isRunning}
                value={trainSize}
                onChange={(e) => setTrainSize(Number(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded appearance-none cursor-pointer accent-amber-500 disabled:opacity-50"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-zinc-400">
                <span>Validation Sub-split size:</span>
                <span className="text-zinc-300 font-semibold">{valSize} files</span>
              </div>
              <input
                type="range"
                min="5"
                max="200"
                step="5"
                disabled={isRunning}
                value={valSize}
                onChange={(e) => setValSize(Number(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded appearance-none cursor-pointer accent-amber-500 disabled:opacity-50"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-zinc-400">
                <span>Testing Sub-split size:</span>
                <span className="text-zinc-300 font-semibold">{testSize} files</span>
              </div>
              <input
                type="range"
                min="5"
                max="200"
                step="5"
                disabled={isRunning}
                value={testSize}
                onChange={(e) => setTestSize(Number(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded appearance-none cursor-pointer accent-amber-500 disabled:opacity-50"
              />
            </div>

            <div className="pt-2 border-t border-zinc-900 flex justify-between items-center text-[11px]">
              <span className="text-zinc-500 font-bold">TOTAL EXPORTS:</span>
              <span className="text-emerald-400 font-bold bg-zinc-950 px-2 py-0.5 border border-zinc-900 rounded">
                {trainSize + valSize + testSize} Samples
              </span>
            </div>
          </div>

          <div className="space-y-2">
            {!isRunning ? (
              <button
                onClick={handleStartCompilation}
                className="w-full py-3 bg-amber-500 hover:bg-amber-400 active:bg-amber-550 text-zinc-950 text-xs font-bold font-mono rounded-lg shadow-lg flex items-center justify-center space-x-1.5 transition duration-150"
              >
                <Play className="w-4 h-4 fill-zinc-950" />
                <span>EXECUTE EXPERIMENT ITERATION</span>
              </button>
            ) : (
              <button
                onClick={handleStopCompilation}
                className="w-full py-3 bg-rose-600 hover:bg-rose-550 text-zinc-100 text-xs font-bold font-mono rounded-lg flex items-center justify-center space-x-1.5 transition duration-150"
              >
                <StopCircle className="w-4 h-4" />
                <span>TERMINATE COMPILER RUN</span>
              </button>
            )}

            {finished && (
              <button
                onClick={handleDownloadCsv}
                className="w-full py-2.5 bg-zinc-800 border border-zinc-700 hover:bg-zinc-750 text-zinc-200 text-xs font-semibold font-mono rounded-lg flex items-center justify-center space-x-1.5 transition"
              >
                <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
                <span>DOWNLOAD SIMULATED METADATA (.CSV)</span>
              </button>
            )}
          </div>
        </div>

        {/* CMD Terminal output column */}
        <div className="md:col-span-8 flex flex-col bg-zinc-950 border border-zinc-850 rounded-lg overflow-hidden h-[260px]">
          {/* Header */}
          <div className="bg-zinc-900 px-4 py-2 border-b border-zinc-850 flex justify-between items-center text-[10px] font-mono text-zinc-500">
            <div className="flex items-center space-x-2">
              <Terminal className="w-3.5 h-3.5 text-amber-500/80" />
              <span className="font-semibold text-zinc-400">Terminal Process Standard Outputs</span>
            </div>
            {isRunning && (
              <div className="flex items-center space-x-1">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-ping"></span>
                <span className="text-emerald-400 font-semibold tracking-wider">RUNNING SOLVES</span>
              </div>
            )}
          </div>

          {/* Lines */}
          <div className="flex-1 p-3.5 overflow-y-auto font-mono text-[11px] space-y-1.5 select-text">
            {logLines.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-650 text-center select-none py-10">
                <Terminal className="w-8 h-8 text-zinc-800 mb-2" />
                <p>Pipeline is idle. Configure counts and click "Execute Experiment Iteration".</p>
                <p className="text-[10px] text-zinc-700 mt-1">Numerical models will run asynchronously.</p>
              </div>
            ) : (
              logLines.map((line, idx) => {
                let colorClass = "text-zinc-400";
                if (line.type === "success") colorClass = "text-emerald-400 font-medium";
                else if (line.type === "warning") colorClass = "text-amber-500 font-medium";
                else if (line.type === "error") colorClass = "text-rose-500 font-semibold";
                else if (line.type === "pde") colorClass = "text-indigo-400";

                return (
                  <div key={idx} className={`${colorClass} hover:bg-zinc-900/30 px-1 py-0.5 rounded leading-normal`}>
                    {line.text}
                  </div>
                );
              })
            )}
            <div ref={terminalEndRef} />
          </div>

          {/* Progress bar footer */}
          <div className="bg-zinc-900/40 border-t border-zinc-900 px-4 py-2 flex justify-between items-center text-[10px] font-mono">
            <div className="flex-1 mr-4">
              <div className="w-full bg-zinc-900 h-1 rounded-full overflow-hidden">
                <div 
                  className="bg-amber-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <span className="text-zinc-500 font-bold min-w-[35px] text-right">
              {progress}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
