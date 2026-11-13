import React, { useState, useRef, useEffect } from 'react'
import {
  UploadCloud,
  Info,
  SlidersHorizontal,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { uploadPredictImage } from '../services/api'
import { PredictResponse, SamplePredictionItem } from '../types/api'

interface RealPredictionPageProps {
  samples: SamplePredictionItem[]
}

export function RealPredictionPage({ samples }: RealPredictionPageProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileDetails, setFileDetails] = useState<{ name: string; format: string } | null>(null)
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [invertOption, setInvertOption] = useState<boolean | undefined>(undefined)
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLElement | null>(null)
  const shouldScrollRef = useRef(false)
  const requestIdRef = useRef(0)
  const abortControllerRef = useRef<AbortController | null>(null)

  const getFileFormat = (fileName: string): string => {
    const ext = fileName.slice(fileName.lastIndexOf('.')).toLowerCase()
    if (ext === '.jpg' || ext === '.jpeg') return 'JPEG'
    if (ext === '.png') return 'PNG'
    if (ext === '.webp') return 'WEBP'
    return ext.replace('.', '').toUpperCase() || 'IMAGE'
  }

  // Smoothly scroll to Inference Results once prediction successfully finishes
  useEffect(() => {
    if (prediction && shouldScrollRef.current && resultsRef.current) {
      shouldScrollRef.current = false
      const prefersReduced =
        typeof window !== 'undefined' &&
        window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches

      resultsRef.current.scrollIntoView({
        behavior: prefersReduced ? 'auto' : 'smooth',
        block: 'start',
      })
    }
  }, [prediction])

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0])
    }
    // Reset input value so repeated uploads of the same file trigger change
    e.target.value = ''
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0])
    }
  }

  const processSelectedFile = (file: File) => {
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    const validExtensions = ['.png', '.jpg', '.jpeg', '.webp']

    if (!validExtensions.includes(ext)) {
      setError(
        `Invalid file format "${file.name}". Only PNG, JPG, JPEG, and WEBP images are supported by the CNN inference pipeline.`
      )
      setSelectedFile(null)
      setPreviewUrl(null)
      setFileDetails(null)
      setPrediction(null)
      shouldScrollRef.current = false
      return
    }

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl)
    }

    const url = URL.createObjectURL(file)
    setSelectedFile(file)
    setFileDetails({
      name: file.name,
      format: getFileFormat(file.name),
    })
    setPreviewUrl(url)
    setError(null)
    setPrediction(null)
    shouldScrollRef.current = true

    // Run real inference
    executeInference(file, invertOption)
  }

  const executeInference = async (file: File, invert?: boolean) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const controller = new AbortController()
    abortControllerRef.current = controller
    const currentRequestId = ++requestIdRef.current

    try {
      setLoading(true)
      setError(null)
      const res = await uploadPredictImage(file, invert, controller.signal)
      if (currentRequestId !== requestIdRef.current) {
        return
      }
      setPrediction(res)
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return
      }
      if (currentRequestId !== requestIdRef.current) {
        return
      }
      shouldScrollRef.current = false
      setPrediction(null)
      setError(err.message || 'Model prediction failed. Please ensure the backend is running.')
    } finally {
      if (currentRequestId === requestIdRef.current) {
        setLoading(false)
      }
    }
  }

  const handleSelectSample = async (sample: SamplePredictionItem) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const controller = new AbortController()
    abortControllerRef.current = controller
    const currentRequestId = ++requestIdRef.current

    const fileName = `sample_${sample.test_index}_${sample.true_label_name}.png`

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl)
    }

    // Immediately display preview inside upload section
    setSelectedFile(null)
    setFileDetails({
      name: fileName,
      format: 'PNG',
    })
    setPreviewUrl(sample.image_base64)
    setError(null)
    setPrediction(null)
    shouldScrollRef.current = true

    try {
      setLoading(true)
      const res = await fetch(sample.image_base64, { signal: controller.signal })
      const blob = await res.blob()
      const file = new File([blob], fileName, {
        type: 'image/png',
      })
      if (currentRequestId !== requestIdRef.current) {
        return
      }
      setSelectedFile(file)
      const predRes = await uploadPredictImage(file, false, controller.signal)
      if (currentRequestId !== requestIdRef.current) {
        return
      }
      setPrediction(predRes)
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return
      }
      if (currentRequestId !== requestIdRef.current) {
        return
      }
      shouldScrollRef.current = false
      setPrediction(null)
      setError(err.message || 'Failed to process sample image')
    } finally {
      if (currentRequestId === requestIdRef.current) {
        setLoading(false)
      }
    }
  }

  // Sorted Top 3 Predictions
  const topThree = prediction
    ? [...prediction.probabilities].sort((a, b) => b.probability - a.probability).slice(0, 3)
    : []

  // Full 10-class probability chart data
  const chartData = prediction
    ? prediction.probabilities.map((item) => ({
        name: item.class_name,
        percentage: item.percentage,
        isTop: item.class_id === prediction.predicted_class_id,
      }))
    : []

  return (
    <div className="max-w-5xl mx-auto space-y-16 pb-20">
      {/* Editorial Header */}
      <section className="pt-4 sm:pt-8 border-b border-stone-200 pb-10">
        <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest uppercase text-stone-500 mb-3">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
          <span>Real-Time Inference Console &bull; Keras CNN Service</span>
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-950 tracking-tight leading-tight">
          Live Fashion Item Classification
        </h1>

        <p className="mt-4 text-base text-stone-600 leading-relaxed max-w-3xl font-normal">
          Upload any apparel image or test with genuine Fashion-MNIST test samples. 
          The backend service converts images to single-channel 28&times;28 tensors, normalizes pixel 
          intensities to [0.0, 1.0], executes the real saved CNN, and returns verified softmax probabilities.
        </p>
      </section>

      {/* Prominent Upload Area */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">01</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Input Console &bull; Upload or Select Garment
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">PNG, JPG, JPEG, WEBP</span>
        </div>

        {/* Large Drag & Drop Box */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-indigo-600 bg-indigo-50/50'
              : 'border-stone-300 hover:border-stone-400 bg-white hover:bg-stone-50/40 shadow-xs'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
            onChange={handleFileChange}
            className="hidden"
          />

          {previewUrl ? (
            <div>
              <div className="w-28 h-28 sm:w-32 sm:h-32 mx-auto rounded-lg bg-black border border-stone-300 flex items-center justify-center overflow-hidden shadow-xs mb-4">
                <img
                  src={previewUrl}
                  alt={fileDetails?.name || 'Selected apparel preview'}
                  className="w-full h-full object-contain pixelated"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>

              <div className="flex flex-wrap items-center justify-center gap-2 max-w-md mx-auto">
                <span
                  className="font-mono text-xs sm:text-sm font-semibold text-stone-900 truncate max-w-xs"
                  title={fileDetails?.name}
                >
                  {fileDetails?.name}
                </span>
                {fileDetails?.format && (
                  <span className="text-[10px] font-mono uppercase font-bold px-2 py-0.5 rounded bg-stone-100 text-stone-700 border border-stone-200">
                    {fileDetails.format}
                  </span>
                )}
              </div>

              <p className="text-xs text-stone-500 mt-2 max-w-md mx-auto leading-relaxed">
                {loading
                  ? 'Computing CNN feed-forward inference...'
                  : 'Drop another apparel image here, or click to browse and replace'}
              </p>

              <div className="mt-4 inline-flex items-center gap-2 text-[11px] font-mono text-stone-400 border border-stone-200 bg-white px-3 py-1 rounded-md">
                <span>FastAPI POST /api/predict</span>
                <span>&bull;</span>
                <span>Max 10MB</span>
              </div>
            </div>
          ) : (
            <div>
              <div className="w-12 h-12 rounded-full border border-stone-200 bg-stone-50 flex items-center justify-center mx-auto text-stone-600 mb-4">
                <UploadCloud className="w-6 h-6 text-stone-700" />
              </div>

              <h3 className="text-base font-bold text-stone-950 tracking-tight">
                Drop an apparel image here, or click to browse
              </h3>

              <p className="text-xs text-stone-500 mt-2 max-w-md mx-auto leading-relaxed">
                Upload any apparel photo (PNG, JPG, JPEG, or WEBP). The pipeline will automatically downsample, 
                convert to grayscale, and stream to the trained CNN.
              </p>

              <div className="mt-5 inline-flex items-center gap-2 text-[11px] font-mono text-stone-400 border border-stone-200 bg-white px-3 py-1 rounded-md">
                <span>FastAPI POST /api/predict</span>
                <span>&bull;</span>
                <span>Max 10MB</span>
              </div>
            </div>
          )}
        </div>

        {/* Inversion and Preprocessing Controls */}
        <div className="p-4 rounded-lg bg-white border border-stone-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2 text-stone-700">
            <SlidersHorizontal className="w-4 h-4 text-stone-500 shrink-0" />
            <span className="font-semibold text-stone-900">Background Inversion:</span>
            <span className="text-stone-500">
              Fashion-MNIST expects light garments on dark backgrounds.
            </span>
          </div>

          <div className="flex items-center gap-2 font-mono text-[11px]">
            <button
              onClick={() => {
                setInvertOption(undefined)
                if (selectedFile) {
                  shouldScrollRef.current = true
                  executeInference(selectedFile, undefined)
                }
              }}
              className={`px-3 py-1 rounded border transition-colors ${
                invertOption === undefined
                  ? 'bg-stone-900 text-white border-stone-900 font-semibold'
                  : 'bg-white text-stone-600 border-stone-200 hover:bg-stone-50'
              }`}
            >
              Auto Detect
            </button>
            <button
              onClick={() => {
                setInvertOption(true)
                if (selectedFile) {
                  shouldScrollRef.current = true
                  executeInference(selectedFile, true)
                }
              }}
              className={`px-3 py-1 rounded border transition-colors ${
                invertOption === true
                  ? 'bg-stone-900 text-white border-stone-900 font-semibold'
                  : 'bg-white text-stone-600 border-stone-200 hover:bg-stone-50'
              }`}
            >
              Invert (White BG)
            </button>
            <button
              onClick={() => {
                setInvertOption(false)
                if (selectedFile) {
                  shouldScrollRef.current = true
                  executeInference(selectedFile, false)
                }
              }}
              className={`px-3 py-1 rounded border transition-colors ${
                invertOption === false
                  ? 'bg-stone-900 text-white border-stone-900 font-semibold'
                  : 'bg-white text-stone-600 border-stone-200 hover:bg-stone-50'
              }`}
            >
              Standard (Dark BG)
            </button>
          </div>
        </div>

        {/* Quick Test Samples */}
        {samples.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono uppercase tracking-wider text-stone-400 font-semibold text-[10px]">
                Quick Test Samples &bull; Genuine Held-Out Test Set
              </span>
              <span className="text-stone-400 font-mono text-[11px]">
                Click any garment to run inference
              </span>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
              {samples.slice(0, 8).map((sample) => (
                <button
                  key={sample.test_index}
                  onClick={() => handleSelectSample(sample)}
                  className="p-2 rounded-lg bg-white border border-stone-200 hover:border-indigo-600 hover:shadow-xs transition-all text-center group"
                >
                  <img
                    src={sample.image_base64}
                    alt={sample.true_label_name}
                    className="w-12 h-12 mx-auto object-contain rounded bg-black pixelated"
                    style={{ imageRendering: 'pixelated' }}
                  />
                  <span className="text-[10px] font-mono text-stone-600 group-hover:text-indigo-950 font-medium block mt-1.5 truncate">
                    {sample.true_label_name}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-xs">
          <p className="font-semibold">{error}</p>
        </div>
      )}

      {/* Prediction Output Results (Ranked Layout) */}
      {(prediction || loading) && (
        <section ref={resultsRef} className="space-y-6">
          <div className="flex items-center justify-between border-b border-stone-200 pb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold text-stone-400">02</span>
              <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
                Inference Results &bull; Top-3 Ranked Probabilities
              </h2>
            </div>
            {loading && (
              <span className="text-[11px] font-mono text-indigo-600 animate-pulse">
                Computing CNN Feed-Forward Pass...
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left: Input Image Inspection */}
            <div className="p-6 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
              <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                Preprocessed Input
              </span>

              {previewUrl && (
                <div className="w-36 h-36 mx-auto rounded-lg bg-black border border-stone-300 flex items-center justify-center overflow-hidden">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-full object-contain pixelated"
                    style={{ imageRendering: 'pixelated' }}
                  />
                </div>
              )}

              <div className="pt-2 border-t border-stone-100 space-y-2 text-xs font-mono text-stone-600">
                <div className="flex justify-between">
                  <span className="text-stone-400">Input Tensor:</span>
                  <span className="font-semibold text-stone-800">1 &times; 28 &times; 28 &times; 1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-stone-400">Target Resolution:</span>
                  <span className="text-stone-800">28 &times; 28 px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-stone-400">Dynamic Range:</span>
                  <span className="text-stone-800">[0.0, 1.0] float32</span>
                </div>
              </div>
            </div>

            {/* Right: Ranked Result Layout */}
            <div className="md:col-span-2 p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-6">
              {prediction ? (
                <>
                  {/* Large Predicted Class Header */}
                  <div className="border-b border-stone-100 pb-5">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                      Predicted Category
                    </span>
                    <div className="flex flex-wrap items-baseline gap-3 mt-1">
                      <h3 className="text-3xl sm:text-4xl font-extrabold text-stone-950 tracking-tight">
                        {prediction.predicted_class_name}
                      </h3>
                      <span className="text-xs font-mono font-bold text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded border border-indigo-100">
                        {prediction.confidence_percentage}% Confidence
                      </span>
                    </div>
                  </div>

                  {/* Refined Top-Three Probability Bars */}
                  <div className="space-y-3">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                      Top-3 Ranked Probabilities
                    </span>

                    <div className="space-y-2.5">
                      {topThree.map((item, idx) => (
                        <div key={item.class_id} className="space-y-1">
                          <div className="flex items-center justify-between text-xs font-mono">
                            <div className="flex items-center gap-2">
                              <span className="text-stone-400 font-bold">#{idx + 1}</span>
                              <span className="font-sans font-semibold text-stone-900">
                                {item.class_name}
                              </span>
                            </div>
                            <span className="text-stone-700 font-bold">{item.percentage}%</span>
                          </div>

                          <div className="h-2 rounded-full bg-stone-100 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                idx === 0 ? 'bg-indigo-600' : 'bg-stone-400'
                              }`}
                              style={{ width: `${Math.max(item.percentage, 1)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Full 10-Class Horizontal Bar Breakdown */}
                  <div className="pt-4 border-t border-stone-100 space-y-2">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-stone-400 block">
                      Full 10-Class Posterior Distribution
                    </span>

                    <div className="h-44 w-full pt-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData}
                          layout="vertical"
                          margin={{ top: 0, right: 30, left: 45, bottom: 0 }}
                        >
                          <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 9, fill: '#78716C' }} />
                          <YAxis
                            dataKey="name"
                            type="category"
                            tick={{ fontSize: 10, fill: '#44403C' }}
                            width={75}
                          />
                          <Tooltip
                            formatter={(val: any) => [`${val}%`, 'Probability']}
                            contentStyle={{
                              backgroundColor: '#1C1917',
                              borderRadius: '4px',
                              border: 'none',
                              color: '#FAF9F6',
                              fontSize: '11px',
                              fontFamily: 'monospace',
                            }}
                          />
                          <Bar dataKey="percentage" radius={[0, 4, 4, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={entry.isTop ? '#4F46E5' : '#D9D5CC'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-12 text-center text-stone-400 text-xs font-mono">
                  Loading prediction...
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Editorial Domain Generalization Caveat */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-stone-200 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-stone-400">03</span>
            <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-stone-900">
              Domain Adaptation Caveat &bull; Fashion-MNIST vs. Real Product Photos
            </h2>
          </div>
          <span className="text-[11px] font-mono text-stone-400">Methodological Note</span>
        </div>

        <div className="p-6 sm:p-8 rounded-lg bg-white border border-stone-200 shadow-xs space-y-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-indigo-600 mt-0.5 shrink-0" />
            <div>
              <h3 className="text-base font-bold text-stone-900 tracking-tight">
                Distribution Shift in E-Commerce Imagery
              </h3>
              <p className="text-xs text-stone-500 mt-0.5">
                Why standard high-resolution product photography may exhibit lower empirical accuracy
              </p>
            </div>
          </div>

          <p className="text-xs sm:text-sm text-stone-600 leading-relaxed font-normal">
            This CNN model is trained strictly on 28&times;28 single-channel images from the 
            canonical <strong>Fashion-MNIST</strong> benchmark dataset. In Fashion-MNIST, items are tightly centered 
            against black backgrounds with standardized normalization. Real-world retail catalog photos 
            differ substantially: they feature varied studio lighting, white backgrounds, human mannequins, 
            and complex multi-color patterns. For production retail deployment, transfer learning with 
            pre-trained RGB backbones (such as EfficientNet or ResNet) on 224&times;224 resolution is recommended.
          </p>
        </div>
      </section>
    </div>
  )
}
