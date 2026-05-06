import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Eraser, Brain, Send, RefreshCw, PenTool, Info } from 'lucide-react';

export default function App() {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Standard MNIST size for internal processing, but displayed larger
    canvas.width = 280;
    canvas.height = 280;

    const context = canvas.getContext('2d');
    if (!context) return;

    context.lineCap = 'round';
    context.strokeStyle = 'white';
    context.lineWidth = 20;
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height); // Initial black background
    contextRef.current = context;
  }, []);

  const startDrawing = ({ nativeEvent }: React.MouseEvent | React.TouchEvent) => {
    const { offsetX, offsetY } = getCoordinates(nativeEvent);
    contextRef.current?.beginPath();
    contextRef.current?.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = ({ nativeEvent }: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(nativeEvent);
    contextRef.current?.lineTo(offsetX, offsetY);
    contextRef.current?.stroke();
  };

  const stopDrawing = () => {
    contextRef.current?.closePath();
    setIsDrawing(false);
  };

  const getCoordinates = (event: any) => {
    if (event.touches) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return { offsetX: 0, offsetY: 0 };
      return {
        offsetX: event.touches[0].clientX - rect.left,
        offsetY: event.touches[0].clientY - rect.top,
      };
    }
    return { offsetX: event.offsetX, offsetY: event.offsetY };
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;
    contextRef.current.fillStyle = 'black';
    contextRef.current.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
  };

  const getPixelData = () => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    // Create a temporary 28x28 canvas to downsample
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return null;

    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28).data;

    // Convert RGBA to grayscale (just take the red channel or average)
    // MNIST is typically normalized 0-1
    const grayscale = [];
    for (let i = 0; i < imageData.length; i += 4) {
      grayscale.push(imageData[i] / 255.0);
    }
    return grayscale;
  };

  const handlePredict = async () => {
    const pixelData = getPixelData();
    if (!pixelData) return;

    setIsPredicting(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: pixelData }),
      });

      const data = await response.json();
      if (data.prediction !== undefined) {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
      }
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4 font-sans">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-white rounded-3xl shadow-xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-indigo-600 p-8 text-white">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-indigo-500 rounded-lg">
              <Brain size={24} />
            </div>
            <h1 className="text-2xl font-bold tracking-tight">MNIST Classifier</h1>
          </div>
          <p className="text-indigo-100 text-sm">Draw a digit (0-9) to see the neural network's prediction.</p>
        </div>

        {/* Canvas Area */}
        <div className="p-6 flex flex-col items-center">
          <div className="relative group">
            <canvas
              id="mnist-canvas"
              ref={canvasRef}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
              className="bg-black rounded-xl cursor-crosshair shadow-inner ring-4 ring-slate-100 group-hover:ring-indigo-100 transition-all duration-300"
              style={{ touchAction: 'none' }}
            />
            {isDrawing && (
              <div className="absolute top-2 right-2 flex items-center gap-2 bg-indigo-600 text-white px-2 py-1 rounded-full text-xs font-medium animate-pulse">
                <PenTool size={12} />
                Drawing...
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex gap-4 mt-8 w-full">
            <button
              onClick={clearCanvas}
              className="flex-1 flex items-center justify-center gap-2 bg-slate-100 hover:bg-slate-200 text-slate-700 py-3 rounded-xl transition-colors font-medium border border-slate-200"
            >
              <Eraser size={18} />
              Clear
            </button>
            <button
              onClick={handlePredict}
              disabled={isPredicting}
              className="flex-2 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white py-3 rounded-xl transition-all shadow-md shadow-indigo-100 font-semibold"
            >
              {isPredicting ? (
                <RefreshCw size={18} className="animate-spin" />
              ) : (
                <Send size={18} />
              )}
              {isPredicting ? 'Analyzing...' : 'Classify'}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-slate-50 p-6 border-t border-slate-100">
          <AnimatePresence mode="wait">
            {prediction !== null ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex items-center justify-between"
              >
                <div>
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-1">Prediction</span>
                  <div className="text-5xl font-black text-indigo-600 leading-none">{prediction}</div>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-1">Confidence</span>
                  <div className="text-xl font-bold text-slate-700">
                    {(confidence! * 100).toFixed(1)}%
                  </div>
                  <div className="w-32 h-2 bg-slate-200 rounded-full mt-2 overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence! * 100}%` }}
                      className="h-full bg-indigo-500" 
                    />
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col items-center py-4 text-slate-400"
              >
                <Info size={32} strokeWidth={1.5} className="mb-2 opacity-50" />
                <p className="text-sm">Enter a digit to analyze</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Stats/Architecture Info */}
      <div className="mt-8 grid grid-cols-2 gap-4 w-full max-w-md">
        <div className="bg-white/50 backdrop-blur-sm p-4 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Architecture</span>
          <span className="text-sm font-semibold text-slate-700">Neural Network</span>
        </div>
        <div className="bg-white/50 backdrop-blur-sm p-4 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Environment</span>
          <span className="text-sm font-semibold text-slate-700">Sklearn-Like</span>
        </div>
      </div>
    </div>
  );
}
