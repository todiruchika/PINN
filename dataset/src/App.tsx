/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { 
  Sparkles, Terminal, BookOpen, Layers, Play, Info, Cpu, Code2, 
  HelpCircle, ChevronRight, Binary, Waves, LibraryBig
} from "lucide-react";
import Simulator from "./components/Simulator";
import Console from "./components/Console";
import FileViewer from "./components/FileViewer";

export default function App() {
  const [activeTab, setActiveTab] = useState<"simulator" | "console" | "code">("simulator");

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans select-none" id="applet-container">
      
      {/* Upper Subtle Border Strip */}
      <div className="h-1 bg-gradient-to-r from-amber-500 via-orange-500 to-rose-600 w-full" />

      {/* Main Container */}
      <div className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col space-y-8">
        
        {/* Academic Header */}
        <header className="border-b border-zinc-900 pb-6 flex flex-col md:flex-row justify-between items-start md:items-end gap-4" id="academic-header">
          <div>
            <div className="flex items-center space-x-2 text-amber-500 font-mono text-[10px] font-bold tracking-widest uppercase bg-amber-500/5 px-2.5 py-1 border border-amber-500/10 rounded w-fit mb-3">
              <Sparkles className="w-3.5 h-3.5 animate-pulse" />
              <span>Research Dataset Module (PINN Project Suite)</span>
            </div>
            
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-display text-zinc-100 uppercase">
              Semiconductor Thermal Dataset Generator
            </h1>
            
            <p className="text-xs sm:text-sm text-zinc-400 mt-1 max-w-3xl leading-relaxed">
              Solve the steady-state Poisson heat equation <code className="bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800 text-amber-400 font-mono font-semibold text-xs">-k∇²T = Q(x,y)</code> to compile physically rigorous, high-fidelity multidimensional datasets and thermal topologies for Physics-Informed Neural Networks.
            </p>
          </div>

          {/* Core PDE Model Equation Badge box */}
          <div className="bg-zinc-950 border border-zinc-900 p-3.5 rounded-xl flex items-center space-x-4 font-mono text-xs shadow-inner shadow-black shrink-0">
            <div className="border-r border-zinc-850 pr-4 space-y-1">
              <span className="text-[9px] text-zinc-500 block uppercase font-bold tracking-wider">Governing PDE:</span>
              <span className="text-zinc-200 font-bold block">-k ∇²T = Q(x, y)</span>
            </div>
            <div className="space-y-1">
              <span className="text-[9px] text-zinc-500 block uppercase font-bold tracking-wider">Boundary Conditions:</span>
              <span className="text-cyan-400 font-bold block">T = T_ambient  on ∂Ω</span>
            </div>
          </div>
        </header>

        {/* Bento Grid Concept Stats Info */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4" id="pinn-concept-bento">
          <div className="bg-zinc-900/15 border border-zinc-900 hover:border-zinc-850 p-4 rounded-xl transition duration-200 flex space-x-3.5 items-start">
            <div className="p-2.5 bg-amber-505/5 border border-amber-500/10 rounded-lg text-amber-400">
              <Binary className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono">1. Math Modelling (PDE Room)</h4>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-relaxed">
                Continuous models of heat conduction are evaluated by Finite Element discretizations over an $H^1$ Sobolev space, which allows PINNs to map neural weights directly to physical loss residuals.
              </p>
            </div>
          </div>

          <div className="bg-zinc-900/15 border border-zinc-900 hover:border-zinc-850 p-4 rounded-xl transition duration-200 flex space-x-3.5 items-start">
            <div className="p-2.5 bg-emerald-505/5 border border-emerald-500/10 rounded-lg text-emerald-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono">2. Stochastic Layout Geometry</h4>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-relaxed">
                Applies stochastically randomized placements of 1–10 hardware components (CPUs, GPUs, Cache, Memory Controllers), exposing neural models to varied, real-world thermal source patterns.
              </p>
            </div>
          </div>

          <div className="bg-zinc-900/15 border border-zinc-900 hover:border-zinc-850 p-4 rounded-xl transition duration-200 flex space-x-3.5 items-start">
            <div className="p-2.5 bg-indigo-505/5 border border-indigo-500/10 rounded-lg text-indigo-400">
              <Waves className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider font-mono">3. Sparse Sensor Gradients</h4>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-relaxed">
                Generates coordinate arrays with 5, 8, and 10 sparse sensors. Excellent for validating sparse regression reconstructions, temperature mappings, and real-time inference checks.
              </p>
            </div>
          </div>
        </section>

        {/* Tab Selector Navigation */}
        <section className="flex border-b border-zinc-900" id="pipeline-navigation-tabs">
          <button
            onClick={() => setActiveTab("simulator")}
            className={`px-5 py-3.5 text-xs font-semibold font-mono tracking-wider uppercase transition border-b-2 flex items-center space-x-2 ${
              activeTab === "simulator"
                ? "border-amber-500 text-amber-400 bg-amber-500/5 font-bold"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>Interactive Simulator Playground</span>
          </button>
          
          <button
            onClick={() => setActiveTab("console")}
            className={`px-5 py-3.5 text-xs font-semibold font-mono tracking-wider uppercase transition border-b-2 flex items-center space-x-2 ${
              activeTab === "console"
                ? "border-amber-500 text-amber-400 bg-amber-500/5 font-bold"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Terminal className="w-4 h-4" />
            <span>Multi-threaded dataset Console</span>
          </button>

          <button
            onClick={() => setActiveTab("code")}
            className={`px-5 py-3.5 text-xs font-semibold font-mono tracking-wider uppercase transition border-b-2 flex items-center space-x-2 ${
              activeTab === "code"
                ? "border-amber-500 text-amber-400 bg-amber-500/5 font-bold"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Code2 className="w-4 h-4" />
            <span>Python deliverables Explorer</span>
          </button>
        </section>

        {/* Dynamic content routing */}
        <main className="flex-1 min-h-[500px]" id="tab-content-area">
          {activeTab === "simulator" && (
            <div className="animate-fade-in transition duration-300">
              <Simulator />
            </div>
          )}
          {activeTab === "console" && (
            <div className="animate-fade-in transition duration-300">
              <Console />
            </div>
          )}
          {activeTab === "code" && (
            <div className="animate-fade-in transition duration-300">
              <FileViewer />
            </div>
          )}
        </main>

        {/* Physics-Informed Neural Network (PINN) Research Info Panel */}
        <section className="bg-zinc-950 border border-zinc-900 rounded-xl p-5 flex flex-col md:flex-row gap-5 items-start md:items-center justify-between" id="pinn-explanation-box">
          <div className="space-y-1">
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-widest font-mono flex items-center space-x-1.5">
              <LibraryBig className="w-4 h-4 text-amber-500" />
              <span>Physics-Informed Deep Learning Integration Guidelines</span>
            </h4>
            <p className="text-[11px] text-zinc-400 leading-relaxed max-w-4xl">
              When training a PINN model (e.g. using PyTorch or TensorFlow) on this dataset, the raw matrices are used to satisfy standard data-driven loss, while the boundary temperature <code className="bg-zinc-900 px-1 py-0.5 rounded text-amber-400">T_amb</code> and heat source map <code className="bg-zinc-900 px-1 py-0.5 rounded text-amber-400">Q(x,y)</code> formulate the exact physical residual equations. Your neural network takes spatial coordinates <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-300">(x, y)</code> as continuous inputs and infers predicted temperature <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-300">T_pred</code>, minimizing the custom differential loss penalty field:
              <br />
              <code className="text-[11px] font-bold text-amber-500 block mt-2 text-center bg-zinc-900/50 p-2 border border-zinc-900 rounded select-text max-w-fit mx-auto">
                Loss_Physics = MeanSquareError( -k * (d²T_pred/dx² + d²T_pred/dy²) - Q(x,y) )
              </code>
            </p>
          </div>
        </section>

      </div>

      {/* Structured Footer */}
      <footer className="bg-zinc-950 border-t border-zinc-900 py-6 mt-12 text-center text-xs text-zinc-600 font-mono" id="applet-footer">
        <div className="max-w-7xl mx-auto px-4 space-y-1.5 select-text">
          <p>© 2026 PINN Thermal Dataset Generation Core. All physical models resolved under isotropic Laplace formulations.</p>
          <p className="text-[10px] text-zinc-700">Powered by the finite-element solving framework FEniCS/DOLFINx and parallel multiprocess scheduling.</p>
        </div>
      </footer>

    </div>
  );
}
