/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { 
  Play, Cpu, RefreshCw, Thermometer, Info, Eye, Layers, Settings, Radio 
} from "lucide-react";
import { FloorplanBlock, SensorConfig, SensorReadingsGroup, SimulationResult } from "../types";
import { 
  generateRandomFloorplan, solveHeatEquation, extractSensorsGroup, getColormapColor 
} from "../utils/solver";

interface Preset {
  name: string;
  description: string;
  blocks: FloorplanBlock[];
}

const CHIP_PRESETS: Preset[] = [
  {
    name: "Apple-Style SoC Layout",
    description: "Multi-core layout with split CPU cores, integrated GPU, system level cache, and memory controllers.",
    blocks: [
      { id: "cpu_0", type: "CPU", x_min: 0.1, y_min: 0.5, x_max: 0.35, y_max: 0.8, width: 0.25, height: 0.3, power_density: 2.2e7 },
      { id: "cpu_1", type: "CPU", x_min: 0.1, y_min: 0.15, x_max: 0.35, y_max: 0.45, width: 0.25, height: 0.3, power_density: 2.2e7 },
      { id: "gpu_0", type: "GPU", x_min: 0.45, y_min: 0.4, x_max: 0.9, y_max: 0.85, width: 0.45, height: 0.45, power_density: 1.6e7 },
      { id: "cache_0", type: "Cache", x_min: 0.45, y_min: 0.15, x_max: 0.9, y_max: 0.32, width: 0.45, height: 0.17, power_density: 2e6 },
      { id: "mc_0", type: "MC", x_min: 0.92, y_min: 0.1, x_max: 0.97, y_max: 0.9, width: 0.05, height: 0.8, power_density: 6.5e6 }
    ]
  },
  {
    name: "Modern Bento-Grid GPU",
    description: "Highly symmetrical parallel processor grid packing massive unified vector math blocks and cache.",
    blocks: [
      { id: "gpu_left", type: "GPU", x_min: 0.1, y_min: 0.1, x_max: 0.45, y_max: 0.9, width: 0.35, height: 0.8, power_density: 1.8e7 },
      { id: "gpu_right", type: "GPU", x_min: 0.55, y_min: 0.1, x_max: 0.9, y_max: 0.9, width: 0.35, height: 0.8, power_density: 1.8e7 },
      { id: "shared_cache", type: "Cache", x_min: 0.46, y_min: 0.15, x_max: 0.54, y_max: 0.85, width: 0.08, height: 0.7, power_density: 1.5e6 }
    ]
  },
  {
    name: "Dual-Core Cloud Substrate",
    description: "Symmetrical layout representing compute clusters bounded by peripheral network interfaces.",
    blocks: [
      { id: "core_a", type: "CPU", x_min: 0.15, y_min: 0.25, x_max: 0.45, y_max: 0.75, width: 0.3, height: 0.5, power_density: 2.4e7 },
      { id: "core_b", type: "CPU", x_min: 0.55, y_min: 0.25, x_max: 0.85, y_max: 0.75, width: 0.3, height: 0.5, power_density: 2.4e7 },
      { id: "mc_top", type: "MC", x_min: 0.1, y_min: 0.85, x_max: 0.9, y_max: 0.93, width: 0.8, height: 0.08, power_density: 7e6 },
      { id: "mc_bottom", type: "MC", x_min: 0.1, y_min: 0.07, x_max: 0.9, y_max: 0.15, width: 0.8, height: 0.08, power_density: 7e6 }
    ]
  }
];

export default function Simulator() {
  const [k, setK] = useState<number>(120);
  const [ambient, setAmbient] = useState<number>(30);
  const [activePresetIndex, setActivePresetIndex] = useState<number>(0);
  const [blocks, setBlocks] = useState<FloorplanBlock[]>(CHIP_PRESETS[0].blocks);
  const [gridSize] = useState<number>(128); // Standard grid matches DOLFINx dataset grid
  const [colormap, setColormap] = useState<"inferno" | "jet">("inferno");
  const [sensorDensity, setSensorDensity] = useState<5 | 8 | 10>(5);
  
  // Simulation results
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  
  // Probe details
  const [probe, setProbe] = useState<{
    x: number;
    y: number;
    val: number;
    blockType: string | null;
    gridX: number;
    gridY: number;
  } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Solves steady-state system on demand
  const triggerSimulationSolve = () => {
    const solved = solveHeatEquation(k, ambient, blocks, gridSize);
    const sensorSets = extractSensorsGroup(solved.grid, gridSize);
    
    setSimResult({
      sample_id: "SIM_LIVE_PREVIEW",
      blocks,
      grid: solved.grid,
      min_temp: solved.min,
      max_temp: solved.max,
      sensors: sensorSets,
      k,
      ambient
    });
  };

  // Run when solver variables change
  useEffect(() => {
    triggerSimulationSolve();
  }, [k, ambient, blocks, colormap]);

  // Handle Preset selection
  const handleApplyPreset = (index: number) => {
    setActivePresetIndex(index);
    setBlocks(CHIP_PRESETS[index].blocks);
  };

  // Randomized manual chip layout triggers
  const handleRandomizeLayout = () => {
    setActivePresetIndex(-1); // Custom tag
    const randomBlocks = generateRandomFloorplan(4 + Math.floor(Math.random() * 5));
    setBlocks(randomBlocks);
  };

  // Draw simulation onto canvas
  useEffect(() => {
    if (!simResult || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const grid = simResult.grid;
    const N = grid.length;
    
    // Create image buffer for fast vector painting
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    const subX = canvas.width / N;
    const subY = canvas.height / N;

    // Draw grid pixels
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const temp = grid[i][j];
        const colorStr = getColormapColor(temp, simResult.min_temp, simResult.max_temp, colormap);
        
        // Parse rgb value
        const rgb = colorStr.match(/\d+/g)?.map(Number) || [0, 0, 0];
        
        // Fill canvas area (origin lower matches Cartesian layouts)
        // Flip Y coords
        for (let py = 0; py < subY; py++) {
          const cy = canvas.height - 1 - (i * subY + py);
          if (cy < 0 || cy >= canvas.height) continue;
          
          for (let px = 0; px < subX; px++) {
            const cx = Math.floor(j * subX + px);
            if (cx < 0 || cx >= canvas.width) continue;
            
            const pixelIdx = (cy * canvas.width + cx) * 4;
            imgData.data[pixelIdx] = rgb[0];     // Red
            imgData.data[pixelIdx + 1] = rgb[1]; // Green
            imgData.data[pixelIdx + 2] = rgb[2]; // Blue
            imgData.data[pixelIdx + 3] = 255;    // Alfa
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Overlay hardware wireframe borders transparently
    blocks.forEach((rect) => {
      ctx.strokeStyle = "rgba(45, 212, 191, 0.45)"; // Teal bordered logic
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);

      const cx = rect.x_min * canvas.width;
      const cy = canvas.height - (rect.y_max * canvas.height); // Correct top-left canvas conversion
      const cw = rect.width * canvas.width;
      const ch = rect.height * canvas.height;

      ctx.strokeRect(cx, cy, cw, ch);
      
      // Paint text block minimally
      ctx.fillStyle = "rgba(45, 212, 191, 0.85)";
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillText(rect.type, cx + 5, cy + 12);
    });

    // Draw active dense sensor locations
    const sensorsKey = `sensors_${sensorDensity}` as keyof SensorReadingsGroup;
    const sensorsConfig = simResult.sensors[sensorsKey];
    
    ctx.setLineDash([]); // Reset dashed lines
    sensorsConfig.positions.forEach((pos, sidx) => {
      const sx = pos[0] * canvas.width;
      const sy = canvas.height - (pos[1] * canvas.height);
      const measuredTemp = sensorsConfig.temperatures?.[sidx] || 30;

      // Draw outer pulsing ripple
      ctx.strokeStyle = "rgba(34, 197, 94, 0.6)"; // lime green
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
      ctx.stroke();

      // Core dot
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, 2 * Math.PI);
      ctx.fill();

      // Border bounds
      ctx.strokeStyle = "#15803d";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

  }, [simResult, blocks, sensorDensity, colormap]);

  // Thermal Probe cursor tracking
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!simResult || !canvasRef.current) return;
    const canvas = canvasRef.current;
    
    const bounds = canvas.getBoundingClientRect();
    const cursorX = e.clientX - bounds.left;
    const cursorY = e.clientY - bounds.top;

    // Convert pixel to standard coordinate spaces [0.0, 1.0]
    const normX = cursorX / canvas.width;
    const normY = 1.0 - (cursorY / canvas.height); // Swap Y direction

    const idxX = Math.max(0, Math.min(gridSize - 1, Math.floor(normX * gridSize)));
    const idxY = Math.max(0, Math.min(gridSize - 1, Math.floor(normY * gridSize)));

    // Fetch localized solver temperature
    const tempCelsius = simResult.grid[idxY][idxX];

    // Detect inside which logic block
    let activeBlockType: string | null = null;
    for (const b of blocks) {
      if (normX >= b.x_min && normX <= b.x_max && normY >= b.y_min && normY <= b.y_max) {
        activeBlockType = b.type;
        break;
      }
    }

    setProbe({
      x: Number(normX.toFixed(3)),
      y: Number(normY.toFixed(3)),
      gridX: idxX,
      gridY: idxY,
      val: Number(tempCelsius.toFixed(2)),
      blockType: activeBlockType
    });
  };

  const handleCanvasMouseLeave = () => {
    setProbe(null);
  };

  // Fetch current active sensor configuration
  const activeSensors = simResult ? simResult.sensors[`sensors_${sensorDensity}` as keyof SensorReadingsGroup] : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="thermography-playground">
      {/* Left controls column */}
      <div className="lg:col-span-5 space-y-5">
        
        {/* presets panel */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Cpu className="w-4 h-4 text-amber-500" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
              Silicon Floorplan Architectures
            </h3>
          </div>

          <div className="grid grid-cols-1 gap-2">
            {CHIP_PRESETS.map((p, idx) => {
              const isSelected = activePresetIndex === idx;
              return (
                <button
                  key={idx}
                  onClick={() => handleApplyPreset(idx)}
                  className={`w-full text-left p-3 rounded-lg border text-xs transition duration-150 ${
                    isSelected
                      ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                      : "bg-zinc-950/40 border-zinc-850 hover:bg-zinc-850 text-zinc-400"
                  }`}
                >
                  <p className="font-semibold font-display">{p.name}</p>
                  <p className="text-[10px] text-zinc-500 mt-0.5 leading-snug">
                    {p.description}
                  </p>
                </button>
              );
            })}
          </div>

          <button
            onClick={handleRandomizeLayout}
            className="w-full mt-3 py-2.5 bg-zinc-800 hover:bg-zinc-700 active:bg-zinc-750 text-zinc-200 border border-zinc-700 text-xs font-semibold rounded-lg font-mono flex items-center justify-center space-x-1.5 transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Generate Randomized Layout (1–10 Blocks)</span>
          </button>
        </div>

        {/* Sliders adjustments */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 space-y-4">
          <div className="flex items-center space-x-2">
            <Settings className="w-4 h-4 text-amber-500" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
              Physical & Mathematical Bounds
            </h3>
          </div>

          {/* Thermal conductivity slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center space-x-1">
                <span className="text-zinc-400">Thermal Conductivity</span>
                <span className="font-mono text-amber-500 italic">k</span>
              </div>
              <span className="font-mono text-zinc-300 bg-zinc-950 border border-zinc-800 px-2 py-0.5 rounded">
                {k} W/(m·K)
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="200"
              step="5"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <p className="text-[10px] text-zinc-500">
              Silicon active heat dispersion constant. Higher means faster convergence, flatter heatmaps.
            </p>
          </div>

          {/* Boundary ambient level */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center space-x-1">
                <span className="text-zinc-400">Boundary Temperature</span>
                <span className="font-mono text-amber-500 italic">T_amb</span>
              </div>
              <span className="font-mono text-zinc-300 bg-zinc-950 border border-zinc-800 px-2 py-0.5 rounded">
                {ambient.toFixed(1)} °C
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="50"
              step="1"
              value={ambient}
              onChange={(e) => setAmbient(Number(e.target.value))}
              className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <p className="text-[10px] text-zinc-500">
              Dirichlet continuous bounds modeling localized chip substrate perimeter cooling temperature.
            </p>
          </div>

          {/* Colormap Select & Grid Select */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="space-y-1.5">
              <span className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">
                Vis Colormap
              </span>
              <div className="flex bg-zinc-950 p-1 border border-zinc-800 rounded-lg text-xs font-mono">
                <button
                  onClick={() => setColormap("inferno")}
                  className={`flex-1 py-1 text-center rounded transition ${
                    colormap === "inferno"
                      ? "bg-zinc-800 text-amber-400 font-bold"
                      : "text-zinc-500 hover:text-zinc-350"
                  }`}
                >
                  inferno
                </button>
                <button
                  onClick={() => setColormap("jet")}
                  className={`flex-1 py-1 text-center rounded transition ${
                    colormap === "jet"
                      ? "bg-zinc-800 text-cyan-400 font-bold"
                      : "text-zinc-500 hover:text-zinc-350"
                  }`}
                >
                  jet
                </button>
              </div>
            </div>

            <div className="space-y-1.5">
              <span className="text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">
                Sensor Density
              </span>
              <div className="flex bg-zinc-950 p-1 border border-zinc-800 rounded-lg text-xs font-mono">
                {([5, 8, 10] as const).map((density) => (
                  <button
                    key={density}
                    onClick={() => setSensorDensity(density)}
                    className={`flex-1 py-1 text-center rounded transition ${
                      sensorDensity === density
                        ? "bg-zinc-800 text-lime-400 font-bold"
                        : "text-zinc-500 hover:text-zinc-350"
                    }`}
                  >
                    {density}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* active block table bounds */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-teal-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                Discrete Substrate Blocks
              </h3>
            </div>
            <span className="text-[11px] font-mono text-zinc-500 bg-zinc-950 border border-zinc-800 px-2 py-0.5 rounded">
              Count: {blocks.length}
            </span>
          </div>

          <div className="max-h-[160px] overflow-y-auto space-y-1.5 pr-1 font-mono text-[10px]">
            {blocks.map((b, bidx) => (
              <div
                key={bidx}
                className="bg-zinc-950/50 border border-zinc-850 p-2.5 rounded-lg flex justify-between items-center hover:border-zinc-750 transition"
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-teal-400">{b.type}</span>
                    <span className="text-zinc-500 text-[9px]">ID: {b.id}</span>
                  </div>
                  <div className="text-zinc-500 mt-1 flex space-x-2">
                    <span>X: [{b.x_min.toFixed(2)}, {b.x_max.toFixed(2)}]</span>
                    <span>Y: [{b.y_min.toFixed(2)}, {b.y_max.toFixed(2)}]</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-zinc-300">{(b.power_density / 1e6).toFixed(1)} MW/m²</p>
                  <p className="text-[9px] text-zinc-500">Power Level</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Center Thermograph Preview Column */}
      <div className="lg:col-span-7 flex flex-col space-y-5">
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-5 flex flex-col items-center">
          <div className="w-full flex justify-between items-center mb-4">
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4 text-amber-500" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                High-Resolution Live Thermography
              </h3>
            </div>
            <div className="flex items-center space-x-1 text-[11px] text-zinc-400 font-mono">
              <Info className="w-3.5 h-3.5 text-zinc-500" />
              <span>Hover canvas to probe temperature</span>
            </div>
          </div>

          {/* Interactive display wrap */}
          <div className="relative border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950 p-2 group shadow-inner shadow-black">
            <canvas
              ref={canvasRef}
              width={384}
              height={384}
              onMouseMove={handleCanvasMouseMove}
              onMouseLeave={handleCanvasMouseLeave}
              className="cursor-crosshair rounded shadow-lg transition duration-200"
            />

            {/* Probe tracking overlay tooltip */}
            {probe && simResult && (
              <div 
                className="absolute bg-zinc-950/90 border border-zinc-800 px-3 py-2 rounded shadow-2xl text-[10px] font-mono text-zinc-300 space-y-1 backdrop-blur-md"
                style={{
                  left: `${Math.min(270, Math.max(10, probe.x * 384 - 40))}px`,
                  bottom: `${Math.min(310, Math.max(10, probe.y * 384 + 15))}px`,
                }}
              >
                <div className="flex justify-between items-center font-bold text-amber-500 border-b border-zinc-900 pb-1 mb-1">
                  <span>TEMPO PROBE</span>
                  <Thermometer className="w-3.5 h-3.5" />
                </div>
                <div className="flex justify-between space-x-4">
                  <span className="text-zinc-500">Temperature:</span>
                  <span className="font-bold text-zinc-100">{probe.val} °C</span>
                </div>
                <div className="flex justify-between space-x-4">
                  <span className="text-zinc-500">Coords (X, Y):</span>
                  <span>({probe.x}, {probe.y})</span>
                </div>
                {probe.blockType && (
                  <div className="flex justify-between space-x-4">
                    <span className="text-zinc-500">Hardware Block:</span>
                    <span className="text-teal-400 font-semibold">{probe.blockType}</span>
                  </div>
                )}
                <div className="flex justify-between space-x-4">
                  <span className="text-zinc-500 font-light">Grid Vertex:</span>
                  <span className="text-zinc-400">({probe.gridX}, {probe.gridY})</span>
                </div>
              </div>
            )}
          </div>

          {/* Live Statistics gauges */}
          {simResult && (
            <div className="w-full mt-5 grid grid-cols-3 gap-3">
              <div className="bg-zinc-950/60 border border-zinc-900 rounded-lg p-3 text-center">
                <p className="text-[10px] font-mono uppercase font-semibold text-zinc-500 tracking-wider">
                  Peak Hotspot Temp
                </p>
                <p className="text-xl font-bold font-display text-rose-500 mt-1 leading-none">
                  {simResult.max_temp.toFixed(1)} °C
                </p>
                <p className="text-[9px] text-zinc-600 font-mono mt-1">
                  At active logic core
                </p>
              </div>

              <div className="bg-zinc-950/60 border border-zinc-900 rounded-lg p-3 text-center">
                <p className="text-[10px] font-mono uppercase font-semibold text-zinc-500 tracking-wider">
                  Uniform Low Boundary
                </p>
                <p className="text-xl font-bold font-display text-cyan-400 mt-1 leading-none">
                  {simResult.min_temp.toFixed(1)} °C
                </p>
                <p className="text-[9px] text-zinc-600 font-mono mt-1">
                  At substrate borders
                </p>
              </div>

              <div className="bg-zinc-950/60 border border-zinc-900 rounded-lg p-3 text-center">
                <p className="text-[10px] font-mono uppercase font-semibold text-zinc-500 tracking-wider">
                  Thermal Gradient ΔT
                </p>
                <p className="text-xl font-bold font-display text-amber-500 mt-1 leading-none">
                  {(simResult.max_temp - simResult.min_temp).toFixed(1)} °C
                </p>
                <p className="text-[9px] text-zinc-600 font-mono mt-1">
                  Physically stable flow
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Live Sensor Gauges panel */}
        {activeSensors && (
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Radio className="w-4 h-4 text-lime-500 animate-pulse" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
                Continuous Sparse Sensor Telemetry ({sensorDensity} Configs)
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-2.5">
              {activeSensors.positions.slice(0, 5).map((pos, sidx) => {
                const temp = activeSensors.temperatures?.[sidx] || 0;
                // Normalize for thermometer bar
                const pct = simResult
                  ? Math.max(0, Math.min(100, ((temp - simResult.min_temp) / (simResult.max_temp - simResult.min_temp)) * 100))
                  : 0;

                return (
                  <div key={sidx} className="bg-zinc-950/50 border border-zinc-850 p-2.5 rounded-lg flex flex-col justify-between font-mono">
                    <div className="flex justify-between text-[9px] text-zinc-500">
                      <span>SEN_{sidx+1}</span>
                      <span>({pos[0].toFixed(2)}, {pos[1].toFixed(2)})</span>
                    </div>
                    <div className="mt-2 text-center">
                      <span className="text-xs font-bold text-lime-400">{temp.toFixed(1)} °C</span>
                    </div>
                    <div className="w-full bg-zinc-900 rounded-full h-1 mt-2 overflow-hidden">
                      <div 
                        className="bg-lime-500 h-full rounded-full transition-all duration-300" 
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              {sensorDensity > 5 && (
                <div className="col-span-1 md:col-span-5 text-center text-[10px] text-zinc-500 bg-zinc-950/30 p-1.5 border border-zinc-900 rounded font-mono">
                  + {sensorDensity - 5} other background sensors actively tracking localized heat expansion blocks. View export files for full matrix coordinates.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
